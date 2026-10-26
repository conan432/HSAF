"""EviTime main line: `gatecalib_use_time`（RankMixer / EST）共享核心。

本文件对应 `说明/evitime_formulation.md` 定稿的主线设计：

  * evidence：请求级 len / staleness / span（等频长度桶 + log 时间桶）→ h_r；
  * Channel A：h_r 加到非序列 token（`_inject_evidence` / `_inject_evidence_rows`）；
  * Channel B：**逐层 gate MLP**，g_{l,r} = λ·σ(MLP_{g,l}(h_r))；RankMixer 的 gate
    作用在 **token mixing 内部**（M-A：`LowRankTokenMixing(x, mix_gate=g_l, hist_slice=...)`
    只缩放历史行，残差保持未门控），EST 的 gate 乘在每层 LCA 读出上；
  * Channel C：逐 task 加性 logit 校准 c_k(h_r)（可选 EMA centering）；
  * time/position score bias：逐层零初始化 key-side bias 加到 attention logits；
  * 训练期：label-free 的 gate-spread hinge（逐层施加后取平均）。

与 `EviTime.py` 的关系（依赖方向）：
    model_zoo/EviTime.py  --(import/继承)-->  EviTime_gatecalib_use_time.EviTimeCoreMixin

即：**本文件是被继承的一方，不 import `EviTime.py`**。消融实验用的开关
（flow_mode / gate_form=scaleshift / aux_loss / film / Plan B,E,F / anchored 等）留在
`EviTime.py`；本文件当前仍包含这些分支的原始实现，下一步会把它们剪掉、只保留主线。

原始文件：model_zoo/EviTime.py（重构前 1–1654 行，逐字搬运）。
"""

# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""EviTime: history-evidence conditioning for the UniRank baselines.

This module ports the evidence machinery of
``TetraAxisNet_dual_memory_evidence`` (see
``说明/TetraAxisNet_dual_memory_evidence_hybfm_user_代码说明.md`` §6-§8) onto
EST / OneTrans / HyFormer / HeMix **without touching those files**:

* three bucketised evidence features (effective length, staleness, span) are
  summed into one ``token_dim`` vector ``evidence``;
* channel A: ``evidence`` is added to every non-sequence token;
* channel B: a per-sample scalar ``g = 2*sigmoid(MLP(evidence))`` scales the
  history-read part of the interaction output (per layer, or - for HeMix -
  the single sequence read performed during tokenization);
* channel C: a per-task additive correction ``calib(evidence)`` is added to
  the raw tower logits before the output activation.

Initialisation follows the reference implementation exactly: small-random
embeddings (``N(0, 0.02^2)``) and small-random *last* layers for the gate and
the calibration head (``N(0, 1e-2^2)`` zero bias), so the variant starts out
behaviourally identical to its baseline while still receiving gradients.

Cost: 70,085 extra parameters on any dataset (token_dim=256, 4 tasks) and a
negligible number of extra FLOPs.

Design notes
------------
* ``evidence_axis`` is fixed to ``"all"``: the evidence vector is added to
  every non-sequence token.  Models with field-level token provenance
  (EST/HyFormer) could restrict it to the user-side tokens, but for a uniform
  cross-model comparison the document's recommendation is to inject into all
  NS tokens.
* bucket slot 0 is reserved for "no history at all" for **all three** features:
  the length feature uses it for an empty sequence, staleness and span shift
  every real value up by one slot (``shift=1``).  Without that offset an empty
  history would share its embedding with a behaviour observed in the very same
  second as the target (staleness == 0) and with a single-behaviour user
  (span == 0).
* OneTrans keeps the *sequence* stream untouched: its mixed attention lets the
  non-sequence queries read history keys, so only the history-key contribution
  of the output is gated (rows of the non-sequence queries).  Gating the whole
  residual would also scale the NS<->NS interaction, which has nothing to do
  with "how much history backs this sample".
* HeMix reads the sequence exactly once, inside ``HeMixTokenizer``; the gate is
  therefore applied once there instead of once per layer.
"""

import math
import types

import torch
from torch import nn

from .EST import EST
from .RankMixer import RankMixer


# Effective-history-length bucket edges (identical to the reference model).
EVIDENCE_LEN_EDGES = (0, 1, 2, 3, 5, 6, 9, 12, 16, 21, 29, 40, 54, 73, 99)
NUM_EVIDENCE_LEN_BUCKETS = len(EVIDENCE_LEN_EDGES) + 1  # 16 slots
DEFAULT_EVIDENCE_TIME_BUCKETS = 64
DEFAULT_EVIDENCE_TIME_BASE = 2.0
# The reference implementation always uses all three evidence dimensions; the
# dimension-ablation runs pass a subset such as ``["len"]``.
DEFAULT_EVIDENCE_DIMS = ("len", "staleness", "span")
_EVIDENCE_DIM_ALIASES = {
    "len": "len", "length": "len", "eff_len": "len",
    "staleness": "staleness", "stale": "staleness", "freshness": "staleness",
    "span": "span", "width": "span",
}


def _normalise_evidence_dims(dims):
    """``("len", "span")`` / ``"len+span"`` / ``["length"]`` -> canonical tuple."""
    if dims is None:
        return DEFAULT_EVIDENCE_DIMS
    if isinstance(dims, str):
        parts = [p for p in dims.replace(",", "+").split("+") if p]
    else:
        parts = list(dims)
    out = []
    for part in parts:
        key = _EVIDENCE_DIM_ALIASES.get(str(part).strip().lower())
        if key is None:
            raise ValueError(f"unknown evidence dimension {part!r}; "
                             "use any of len/staleness/span")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("evidence_dims must not be empty")
    # Keep the canonical order so the summed vector is order-independent.
    return tuple(d for d in DEFAULT_EVIDENCE_DIMS if d in out)


def _evidence_time_buckets(values, num_buckets, base=DEFAULT_EVIDENCE_TIME_BASE,
                           shift=0):
    """Log-spaced bucket ids, same convention as the reference model.

    ``shift`` moves every **real** value up by that many slots.  It is used for
    staleness with ``shift=1`` so that slot 0 is reserved for "there is no
    history at all" - the same convention the length feature already follows
    (``len_bucket == 0`` <=> empty sequence, real lengths start at slot 1).
    Without the shift ``staleness == 0`` (a behaviour observed in the very same
    second as the target) and "no history" would share slot 0.

    The result is always clamped into ``[0, num_buckets - 1]`` so the embedding
    table keeps ``num_buckets`` rows (existing checkpoints keep loading); the
    only information this can cost is the merge of the two astronomically large
    buckets at the top end (staleness >= 2**62 s), which real data never reaches.
    """
    values = values.clamp_min(0.0)
    denominator = math.log(max(base, 1.000001))
    buckets = torch.floor(torch.log1p(values) / denominator).long() + int(shift)
    return buckets.clamp_(0, num_buckets - 1)


def _resolve_time_unit(model, time_unit):
    """Timestamp unit used for staleness/span bucketing."""
    if time_unit:
        return str(time_unit).lower()
    dataset_id = str(getattr(model.feature_map, "dataset_id", "")).lower()
    if "kuairand" in dataset_id:
        return "ms"
    return "s"


def _read_evitime_inputs(inputs):
    """History timestamps / target timestamp appended by the collator."""
    timestamp_seq = inputs[4] if len(inputs) > 4 else None
    target_timestamp = inputs[5] if len(inputs) > 5 else None
    return timestamp_seq, target_timestamp


class EviTimeCoreMixin:
    """Shared evidence modules, feature computation and readout hooks."""

    def _init_evitime(self,
                      use_evidence_token=True, use_history_gate=True,
                      use_logit_calibration=True,
                      num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                      evidence_len_edges=None,
                      num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                      evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                      time_unit=None, evidence_dims=DEFAULT_EVIDENCE_DIMS,
                      use_temporal_bias=False, use_position_bias=False,
                      history_gate_scale=2.0, gate_spread_weight=0.0,
                      gate_spread_target=0.15, calib_task_specific=False,
                      center_history_gate=False, center_calibration=False,
                      center_mode="batch", center_ema_momentum=0.99,
                      center_ema_warmup_batches=20,
                      gate_in_mixing=True, per_layer_gate=True,
                      score_bias_num_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                      score_bias_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                      score_bias_num_layers=None, score_bias_max_len=100):
        dim = self.token_dim
        self.use_evidence_token = bool(use_evidence_token)
        self.use_history_gate = bool(use_history_gate)
        self.use_logit_calibration = bool(use_logit_calibration)
        self.gate_in_mixing = bool(gate_in_mixing)
        self.per_layer_gate = bool(per_layer_gate)
        self.evidence_axis = "all"
        if evidence_len_edges is not None:
            edges = [int(e) for e in evidence_len_edges]
            assert edges and edges[0] == 0
            assert all(a < b for a, b in zip(edges, edges[1:]))
            self.evidence_len_edges = tuple(edges)
            self.num_evidence_len_buckets = len(edges)
        else:
            self.evidence_len_edges = tuple(EVIDENCE_LEN_EDGES)
            self.num_evidence_len_buckets = int(num_evidence_len_buckets)
        self.register_buffer("_len_edges_tensor",
                             torch.tensor(self.evidence_len_edges, dtype=torch.float64),
                             persistent=False)
        self.evidence_num_time_buckets = int(num_evidence_time_buckets)
        self.evidence_time_base = float(evidence_time_base)
        self.evidence_time_unit = _resolve_time_unit(self, time_unit)
        self.evidence_dims = _normalise_evidence_dims(evidence_dims)
        self.center_history_gate = bool(center_history_gate)
        self.center_calibration = bool(center_calibration)
        self.center_mode = str(center_mode).strip().lower()
        if self.center_mode not in ("batch", "ema"):
            raise ValueError("center_mode must be 'batch' or 'ema'")
        # EMA momentum of the running centering statistics.  Must be defined
        # here: the main line does not inherit from ``EviTime.py``'s mixin, so
        # ``_update_center_stat`` would otherwise raise ``AttributeError`` as
        # soon as the warm-up batches are over.
        self.center_ema_momentum = float(center_ema_momentum)
        self.center_ema_warmup_batches = int(center_ema_warmup_batches)
        self.gate_spread_weight = float(gate_spread_weight)
        self.gate_spread_target = float(gate_spread_target)
        self.calib_task_specific = bool(calib_task_specific)
        self.history_gate_scale = float(history_gate_scale)
        self._last_history_gate = None
        self.score_bias_num_time_buckets = int(score_bias_num_time_buckets)
        self.score_bias_time_base = float(score_bias_time_base)
        self.score_bias_max_len = int(score_bias_max_len)
        default_layers = int(
            getattr(self, "num_layers", 0)
            or getattr(getattr(self, "unified_interaction_layers", None), "num_layers", 0)
            or 1)
        self.score_bias_num_layers = int(score_bias_num_layers or default_layers)
        self.use_temporal_bias = bool(use_temporal_bias)
        self.use_position_bias = bool(use_position_bias)
        self.temporal_bias = (
            nn.Parameter(torch.zeros(self.score_bias_num_layers,
                                     self.score_bias_num_time_buckets))
            if self.use_temporal_bias else None)
        self.position_bias = (
            nn.Parameter(torch.zeros(self.score_bias_num_layers, self.score_bias_max_len))
            if self.use_position_bias else None)
        self.register_buffer("_center_step", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        if self.center_mode == "ema":
            self.register_buffer("_evidence_ema_mean", torch.zeros(dim))
            self.register_buffer("_gate_ema_mean", torch.ones(1))
            self.register_buffer("_calibration_ema_mean", torch.zeros(self.num_tasks))
        active_embeds = []
        if "len" in self.evidence_dims:
            self.evidence_len_embed = nn.Embedding(self.num_evidence_len_buckets, dim)
            active_embeds.append(self.evidence_len_embed)
        if "staleness" in self.evidence_dims:
            self.evidence_staleness_embed = nn.Embedding(self.evidence_num_time_buckets, dim)
            active_embeds.append(self.evidence_staleness_embed)
        if "span" in self.evidence_dims:
            self.evidence_span_embed = nn.Embedding(self.evidence_num_time_buckets, dim)
            active_embeds.append(self.evidence_span_embed)
        if not active_embeds:
            raise ValueError("evidence_dims must keep at least one of len/staleness/span")
        for embed in active_embeds:
            nn.init.normal_(embed.weight, std=0.02)
        hidden = max(dim // 4, 16)
        self.evidence_gate_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for _ in range(int(default_layers))])
        for head in self.evidence_gate_layers:
            nn.init.normal_(head[-1].weight, std=1e-2)
            nn.init.zeros_(head[-1].bias)
        if self.calib_task_specific:
            self.evidence_calibration = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                for _ in range(self.num_tasks)])
            for head in self.evidence_calibration:
                nn.init.normal_(head[-1].weight, std=1e-2)
                nn.init.zeros_(head[-1].bias)
        else:
            self.evidence_calibration = nn.Sequential(
                nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, self.num_tasks))
            nn.init.normal_(self.evidence_calibration[-1].weight, std=1e-2)
            nn.init.zeros_(self.evidence_calibration[-1].bias)
        self._last_evidence = None
        self._last_calibration = None
        self._last_length = None

    def _calibration_forward(self, evidence):
        """``(B, num_tasks)`` calibration term (shared trunk or per-task MLPs)."""
        if self.calib_task_specific:
            return torch.cat([head(evidence) for head in self.evidence_calibration],
                             dim=-1)
        return self.evidence_calibration(evidence)

    def _evitime_finalize(self, compile_flag, kwargs):
        """Re-create optimizers (they must own the new parameters)."""
        self._torch_compile_enabled = False
        self.enable_torch_compile = compile_flag
        self.compile(
            kwargs.get("dense_optimizer"),
            kwargs["loss"],
            kwargs.get("dense_learning_rate"),
        )
        self.model_to_device()

    def _sync_center_mean(self, value):
        """Return a rank-local running statistic (main line: EMA centering).

        DDP broadcasts module buffers from rank 0 before later forwards, so a
        plain local update avoids a custom collective inside the forward.
        """
        return value.detach()

    @torch.no_grad()
    def _update_center_stat(self, stat_name, batch_mean, init_value):
        """Update an EMA running mean; the first batches are warm-up means."""
        old = getattr(self, stat_name).detach().clone()
        current = self._sync_center_mean(batch_mean.detach())
        step = int(self._center_step.item())
        if step < self.center_ema_warmup_batches:
            new = current
        else:
            momentum = min(max(self.center_ema_momentum, 0.0), 0.999999)
            new = momentum * old + (1.0 - momentum) * current
        getattr(self, stat_name).copy_(new.to(old.dtype))
        return new

    def _center_tensor(self, tensor, stat_name, dim=None):
        """Center a 2-D tensor by batch mean or the frozen/EMA running mean."""
        if not self.training:
            if self.center_mode != "ema":
                return tensor - tensor.detach().mean(dim=0, keepdim=True)
            mean = getattr(self, stat_name).detach()
            return tensor - mean.reshape([1] + ([-1] if tensor.dim() > 1 else []))
        batch_mean = tensor.detach().mean(dim=0)
        if self.center_mode == "batch":
            return tensor - batch_mean
        # EMA update and centering use the same statistic.  The first few steps
        # use the synchronized batch mean, then momentum takes over.
        self._update_center_stat(stat_name, batch_mean, None)
        mean = getattr(self, stat_name).detach()
        return tensor - mean.reshape([1] + ([-1] if tensor.dim() > 1 else []))

    def _center_scalar(self, tensor, stat_name):
        """Center a scalar by batch mean or EMA; used for history gate."""
        if not self.training:
            if self.center_mode != "ema":
                return tensor / tensor.detach().mean().clamp_min(1e-6)
            return tensor / getattr(self, stat_name).detach().clamp_min(1e-6)
        batch_mean = tensor.detach().mean()
        if self.center_mode == "batch":
            return tensor / batch_mean.clamp_min(1e-6)
        self._update_center_stat(stat_name, batch_mean.reshape(()), torch.ones_like(batch_mean))
        return tensor / getattr(self, stat_name).detach().reshape(()).clamp_min(1e-6)

    @torch.no_grad()
    def _evidence_features(self, mask, timestamp_seq, target_timestamp):
        """Bucketised (length, staleness, span) ids for the current batch."""
        batch_size = mask.size(0)
        empty = torch.zeros(batch_size, dtype=torch.long, device=mask.device)
        effective_len = mask.sum(dim=1).long()
        len_bucket = torch.bucketize(
            effective_len.to(self._len_edges_tensor.dtype),
            self._len_edges_tensor,
        ).clamp_(0, self.num_evidence_len_buckets - 1)
        if timestamp_seq is None or timestamp_seq.numel() == 0:
            return len_bucket, empty, empty

        valid = mask.bool()
        stamps = timestamp_seq.to(self.device, dtype=torch.float64)
        has_history = valid.any(dim=1)
        last_ts = torch.where(
            valid, stamps, torch.full_like(stamps, float("-inf"))
        ).max(dim=1).values
        first_ts = torch.where(
            valid, stamps, torch.full_like(stamps, float("inf"))
        ).min(dim=1).values
        last_ts = torch.where(has_history, last_ts, torch.zeros_like(last_ts))
        first_ts = torch.where(has_history, first_ts, torch.zeros_like(first_ts))

        if target_timestamp is not None:
            target = target_timestamp.to(self.device, dtype=torch.float64).view(-1)
            reference = torch.where(target > 0, target, last_ts)
        else:
            reference = last_ts
        staleness = (reference - last_ts).clamp_min(0.0)
        span = (last_ts - first_ts).clamp_min(0.0)

        if self.evidence_time_unit in {"ms", "millisecond", "milliseconds"}:
            scale = 1.0 / 1000.0
        elif self.evidence_time_unit in {"us", "microsecond", "microseconds"}:
            scale = 1.0 / 1000000.0
        else:
            scale = 1.0
        # ``shift=1``: slot 0 is reserved for "no history", real values start at
        # slot 1 (see ``_evidence_time_buckets``).  The samples without history
        # are then pinned to slot 0 explicitly, so an *empty* history can no
        # longer be confused with a very *fresh* one (staleness == 0) nor with a
        # single-behaviour user (span == 0).
        staleness_bucket = _evidence_time_buckets(
            staleness * scale, self.evidence_num_time_buckets,
            self.evidence_time_base, shift=1,
        )
        staleness_bucket = torch.where(
            has_history, staleness_bucket, torch.zeros_like(staleness_bucket)
        )
        span_bucket = _evidence_time_buckets(
            span * scale, self.evidence_num_time_buckets,
            self.evidence_time_base, shift=1,
        )
        span_bucket = torch.where(
            has_history, span_bucket, torch.zeros_like(span_bucket)
        )
        return len_bucket, staleness_bucket, span_bucket

    def _compute_evidence(self, mask, timestamp_seq, target_timestamp):
        self._last_length = mask.sum(dim=1).float().detach()
        len_bucket, staleness_bucket, span_bucket = self._evidence_features(
            mask, timestamp_seq, target_timestamp)
        evidence = None
        if "len" in self.evidence_dims:
            evidence = self.evidence_len_embed(len_bucket)
        if "staleness" in self.evidence_dims:
            term = self.evidence_staleness_embed(staleness_bucket)
            evidence = term if evidence is None else evidence + term
        if "span" in self.evidence_dims:
            term = self.evidence_span_embed(span_bucket)
            evidence = term if evidence is None else evidence + term
        self._last_evidence = evidence
        history_gate = None
        if self.use_history_gate:
            gates = []
            for head in self.evidence_gate_layers:
                g = self.history_gate_scale * torch.sigmoid(head(evidence))
                if self.center_history_gate:
                    g = self._center_scalar(g, "_gate_ema_mean")
                gates.append(g)
            history_gate = torch.stack(gates, dim=0).unsqueeze(-1)
        self._last_history_gate = history_gate
        calibration = (self._calibration_forward(evidence)
                       if self.use_logit_calibration else None)
        if calibration is not None and self.center_calibration:
            calibration = self._center_tensor(calibration, "_calibration_ema_mean")
        if self.training and self.center_mode == "ema":
            with torch.no_grad():
                self._center_step += 1
        self._last_calibration = calibration
        return evidence, history_gate, calibration

    def add_loss(self, return_dict, y_true):
        loss = super().add_loss(return_dict, y_true)
        if not self.training:
            return loss
        gate = self._last_history_gate
        if self.gate_spread_weight > 0 and gate is not None and gate.numel() > 1:
            floor = self.gate_spread_target
            if gate.dim() >= 4:
                penalty = 0.0
                for li in range(gate.size(0)):
                    spread = gate[li].reshape(-1).std()
                    ft = torch.as_tensor(floor, dtype=spread.dtype, device=spread.device)
                    penalty = penalty + torch.relu(ft - spread)
                loss = loss + self.gate_spread_weight * penalty / gate.size(0)
            else:
                spread = gate.reshape(-1).std()
                ft = torch.as_tensor(floor, dtype=spread.dtype, device=spread.device)
                loss = loss + self.gate_spread_weight * torch.relu(ft - spread)
        return loss

    def _inject_evidence(self, ns_tokens, evidence):
        """Channel A: add the evidence vector to every non-sequence token."""
        if not self.use_evidence_token:
            return ns_tokens
        return ns_tokens + evidence.unsqueeze(1)

    @staticmethod
    def _apply_calibration(tower_output, calibration):
        """Channel C: per-task additive correction on the raw logits."""
        if calibration is None:
            return tower_output
        return [
            value + calibration[:, index:index + 1]
            for index, value in enumerate(tower_output)
        ]

    def _inject_evidence_rows(self, tokens, evidence, rows):
        """Channel A restricted to a token-row range (used by the mixers)."""
        if not self.use_evidence_token:
            return tokens
        lo, hi = rows
        return torch.cat([
            tokens[:, :lo, :],
            tokens[:, lo:hi, :] + evidence.unsqueeze(1),
            tokens[:, hi:, :],
        ], dim=1)

    # -- temporal / position score bias -------------------------------------

    def _score_bias_time_scale(self):
        """Timestamp unit -> seconds (same convention as the evidence features)."""
        if self.evidence_time_unit in {"ms", "millisecond", "milliseconds"}:
            return 1.0 / 1000.0
        if self.evidence_time_unit in {"us", "microsecond", "microseconds"}:
            return 1.0 / 1000000.0
        return 1.0

    def _score_bias_time_buckets(self, mask, timestamp_seq, target_timestamp):
        """Per-key log-bucketed time gap to the target (``(B, L)`` or ``None``).

        Mirrors the reference: gaps are computed in float64, converted to
        seconds, then bucketed with ``floor(log1p(gap)/log(base))`` and clamped
        into ``[0, num_time_buckets)``.
        """
        if (self.temporal_bias is None or timestamp_seq is None
                or timestamp_seq.numel() == 0):
            return None
        valid = mask.bool()
        stamps = timestamp_seq.to(self.device, dtype=torch.float64)
        has_history = valid.any(dim=1)
        last_ts = torch.where(
            valid, stamps, torch.full_like(stamps, float("-inf"))
        ).max(dim=1).values
        last_ts = torch.where(has_history, last_ts, torch.zeros_like(last_ts))
        if target_timestamp is not None:
            target = target_timestamp.to(self.device, dtype=torch.float64).view(-1)
            reference = torch.where(target > 0, target, last_ts)
        else:
            reference = last_ts
        gap = (reference.unsqueeze(1) - stamps).clamp_min(0.0) * self._score_bias_time_scale()
        denominator = math.log(max(self.score_bias_time_base, 1.000001))
        buckets = torch.floor(torch.log1p(gap) / denominator).long()
        buckets = buckets.clamp_(0, self.score_bias_num_time_buckets - 1)
        # padded keys carry no bias (their bucket index is meaningless)
        return torch.where(valid, buckets, torch.zeros_like(buckets))

    def _score_biases(self, mask, timestamp_seq, target_timestamp, key_length):
        """Per-layer key-side biases, each ``(B, 1, 1, key_length)``.

        Padded keys get a **zero** bias (their padding semantics stay exactly as
        before); the position part measures the distance from the end of the
        history, so the last valid behaviour always has distance 0.
        """
        if self.temporal_bias is None and self.position_bias is None:
            return None
        time_buckets = self._score_bias_time_buckets(mask, timestamp_seq, target_timestamp)
        valid = mask.bool().float()
        biases = []
        for layer_idx in range(self.score_bias_num_layers):
            bias = None
            if self.temporal_bias is not None and time_buckets is not None:
                bias = self.temporal_bias[layer_idx][time_buckets].float() * valid
            if self.position_bias is not None:
                key_position = torch.arange(key_length, device=mask.device)
                distance = (key_length - 1 - key_position).clamp(
                    0, self.position_bias.size(1) - 1)
                term = self.position_bias[layer_idx][distance].float().view(1, key_length)
                term = term * valid[:, :key_length]
                bias = term if bias is None else bias + term
            biases.append(None if bias is None
                          else bias.view(bias.size(0), 1, 1, bias.size(1)))
        return tuple(biases)

def _mixed_hetero_attention_with_score_bias(self, s_tokens, ns_tokens,
                                            score_bias=None):
    """``MixedHeteroAttention.forward`` (HeMix tokenizer) + key-side bias."""
    B, Ls, D = s_tokens.shape
    _, Lns, _ = ns_tokens.shape
    if Ls == 0:
        return ns_tokens

    q_weight = self.W_q_ns[:Lns]
    q_ns = torch.einsum("bld,ldh->blh", ns_tokens, q_weight)
    k_s = self.W_k_s(s_tokens)
    v_s = self.W_v_s(s_tokens)
    q = q_ns.view(B, Lns, self.num_heads, self.head_dim).transpose(1, 2)
    k = k_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
    v = v_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
    output, _ = self.dot_attention(
        q, k, v, scale=self.head_dim ** 0.5, mask=score_bias,
    )
    output = output.transpose(1, 2).contiguous().view(B, Lns, D)
    return self.W_o(output)


def _est_blocks_forward(self, s_tokens, ns_tokens, mask=None,
                        history_gate=None, score_biases=None):
    gate = 1.0 if history_gate is None else history_gate
    for i in range(self.num_layers):
        # per-layer gates arrive as [L, B, 1, 1]; a plain [B, 1, 1] is shared
        layer_gate = gate
        if torch.is_tensor(gate) and gate.dim() >= 4:
            layer_gate = gate[i]
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        norm_s = self.attention_norms[i](s_tokens)
        norm_ns = self.attention_norms[i](ns_tokens)
        bias = None if score_biases is None else score_biases[i]
        ns_tokens = layer_gate * self.lca_layers[i](
            norm_s, norm_ns, mask, score_bias=bias
        ) + ns_tokens

        norm_s = self.ffn_norms[i](s_tokens)
        norm_ns = self.ffn_norms[i](ns_tokens)
        norm_s, norm_ns = self.ffn_layers[i](norm_s, norm_ns, mask)
        s_tokens = s_tokens + norm_s
        ns_tokens = ns_tokens + norm_ns

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

    return s_tokens, ns_tokens


def _hetero_query_tokenizer_forward(self, feature_embeddings, sequence_emb,
                                    mask=None, score_bias=None):
    """``HeteroQueryTokenizer.forward`` + optional key-side score bias.

    The main line only uses the single-window reader of the aligned mixer
    family (``num_real_token == 0``, RankMixer / UniMixer): one query group
    reads the whole history.  HeMix's long/short-term split stays HeMix's own
    contribution, and the ablation-only readout gate / FiLM forms live in
    ``EviTime.py``.

    ``score_bias`` is the full-history key-side bias (``B x 1 x 1 x L``).
    """
    if self.num_real_token > 0:
        raise NotImplementedError(
            "the EviTime main line implements the single-window history read "
            "only (num_real_token == 0); got num_real_token={}".format(
                self.num_real_token))
    if feature_embeddings.dim() > 2:
        feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
    B = feature_embeddings.size(0)
    ns_tokens = self.token_mlps(feature_embeddings)
    if mask is not None:
        sequence_emb = sequence_emb * mask.unsqueeze(-1).float()
    queries = torch.cat([ns_tokens, self.cls_tokens.expand(B, -1, -1)], dim=1)
    history_tokens = self.mha_full(
        s_tokens=sequence_emb,
        ns_tokens=queries,
        score_bias=score_bias,
    )
    return torch.cat([history_tokens, ns_tokens], dim=1)


def _attach_mixing_gate(block, nonseq_slice, history_slice):
    """M-A: make the gate act *inside* the token-mixing operator.

    Nothing is added to the model: ``forward`` simply hands each layer's gate to
    ``LowRankTokenMixing``, which scales the history rows of that layer's mixing
    input (``x_in = [g * x_hist, x_ns]``).  The residual stream keeps the
    ungated representation, so the gate only re-weights what mixing reads.
    """
    block.nonseq_slice = tuple(nonseq_slice)
    block.history_slice = tuple(history_slice)
    block.forward = types.MethodType(_mixer_forward_with_mixing_gate, block)
    return block


def _mixer_forward_with_mixing_gate(self, x, history_gate=None):
    """Mixer forward whose evidence gate lives inside the mixing operator (M-A)."""
    for i in range(self.num_layers):
        gate = None
        if history_gate is not None:
            if torch.is_tensor(history_gate):
                # [L, B, 1, 1] -> this layer's gate; fewer dims = shared gate
                gate = history_gate[i] if history_gate.dim() >= 4 else history_gate
                if gate.dim() == 3 and gate.size(0) != x.size(0):
                    raise ValueError(
                        "mixing gate must be [L,B,1,1] (per layer) or [B,1,1] "
                        "(shared); got shape {}".format(tuple(gate.shape)))
            else:
                gate = history_gate[i]          # list / tuple of per-layer gates
        x = self.mixer_layers[i](
            self.mixer_norms[i](x), mix_gate=gate, hist_slice=self.history_slice
        ) + x
        x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
    return x


# =========================================================================
# 主线模型（`gatecalib_use_time`）：RankMixer / EST
#   依据 说明/evitime_formulation.md 写死：三通道 + 逐层 gate MLP + use_time；
#   RankMixer 的 gate 在 token mixing 内部（M-A），EST 的 gate 在每层 LCA 读出上。
#   这里没有消融开关（flow_mode / gate_form / aux_loss / film / Plan B,E,F 等）。
# =========================================================================


class RankMixer_EviTime_gatecalib_use_time(EviTimeCoreMixin, RankMixer):
    """RankMixer + evidence(gatecalib) + time/position bias，gate 在 mixing 内部。"""

    def __init__(self, feature_map, model_id="RankMixer_EviTime_gatecalib_use_time",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 use_temporal_bias=True,
                 use_position_bias=True,
                 history_gate_scale=2.0,
                 gate_spread_weight=0.5,
                 gate_spread_target=0.15,
                 center_mode="ema",
                 calib_task_specific=True,
                 center_calibration=True,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            center_history_gate=False,
            center_calibration=center_calibration,
            center_mode=center_mode,
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            calib_task_specific=calib_task_specific,
            gate_spread_weight=gate_spread_weight,
            gate_spread_target=gate_spread_target,
            gate_in_mixing=True,        # M-A：gate 作用在 token mixing 内部
            per_layer_gate=True,        # 每层一个 gate MLP
            history_gate_scale=history_gate_scale,
            # RankMixer 只读一次历史 → time/pos bias 只有一层表
            score_bias_num_layers=1,
            score_bias_max_len=int(kwargs.get("max_len", 100)),
        )
        # token layout: [2N history tokens | N non-sequence tokens]
        self.history_token_slice = (0, 2 * self.num_ns_token)
        self.nonseq_token_slice = (2 * self.num_ns_token, self.num_all_tokens)
        _attach_mixing_gate(
            self.unified_interaction_layers,
            nonseq_slice=self.nonseq_token_slice,
            history_slice=self.history_token_slice,
        )
        self.unified_tokenizer_layer.forward = types.MethodType(
            _hetero_query_tokenizer_forward, self.unified_tokenizer_layer
        )
        for attention in (getattr(self.unified_tokenizer_layer, "mha_full", None),
                          getattr(self.unified_tokenizer_layer, "mha_real", None),
                          getattr(self.unified_tokenizer_layer, "mha_glocal", None)):
            if attention is not None:
                attention.forward = types.MethodType(
                    _mixed_hetero_attention_with_score_bias, attention)
        self._evitime_finalize(compile_flag, kwargs)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_evitime_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]

        evidence, history_gate, calibration = self._compute_evidence(
            mask, timestamp_seq, target_timestamp
        )
        score_bias = None
        if self.temporal_bias is not None or self.position_bias is not None:
            score_bias = self._score_biases(
                mask, timestamp_seq, target_timestamp,
                key_length=sequence_emb.size(1),
            )[0]

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        sequence_tokens = self.item_token_proj(sequence_emb)
        if mask is not None:
            sequence_tokens = sequence_tokens * mask.unsqueeze(-1).float()

        # 历史读出保持未门控（gate 在 mixing 内部生效）
        unified_tokens = self.unified_tokenizer_layer(
            feature_embeddings, sequence_tokens, mask, score_bias=score_bias
        )
        unified_tokens = self._inject_evidence_rows(
            unified_tokens, evidence, self.nonseq_token_slice
        )
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers, unified_tokens, history_gate
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        return {
            "{}_pred".format(labels[i]): self.output_activation[i](v)
            for i, v in enumerate(tower_output)
        }


class EST_EviTime_gatecalib_use_time(EviTimeCoreMixin, EST):
    """EST + evidence(gatecalib) + time/position bias，gate 乘在每层 LCA 读出上。"""

    def __init__(self, feature_map, model_id="EST_EviTime_gatecalib_use_time",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 use_temporal_bias=True,
                 use_position_bias=True,
                 history_gate_scale=2.0,
                 gate_spread_weight=0.5,
                 gate_spread_target=0.15,
                 center_mode="ema",
                 calib_task_specific=True,
                 center_calibration=True,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            center_history_gate=False,
            center_calibration=center_calibration,
            center_mode=center_mode,
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            calib_task_specific=calib_task_specific,
            gate_spread_weight=gate_spread_weight,
            gate_spread_target=gate_spread_target,
            gate_in_mixing=False,       # EST 每层都读历史，gate 直接乘在 LCA 读出上
            per_layer_gate=True,
            history_gate_scale=history_gate_scale,
            score_bias_max_len=int(kwargs.get("max_len", 100)),
        )
        self.unified_interaction_layers.forward = types.MethodType(
            _est_blocks_forward, self.unified_interaction_layers
        )
        for layer in self.unified_interaction_layers.lca_layers:
            layer.forward = types.MethodType(_lca_forward_with_score_bias, layer)
        self._evitime_finalize(compile_flag, kwargs)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_evitime_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]
        s_tokens = self.item_token_proj(sequence_emb)

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        ns_tokens = self.unified_tokenizer_layer(feature_embeddings)

        evidence, history_gate, calibration = self._compute_evidence(
            mask, timestamp_seq, target_timestamp
        )
        ns_tokens = self._inject_evidence(ns_tokens, evidence)

        score_biases = self._score_biases(
            mask, timestamp_seq, target_timestamp, key_length=s_tokens.size(1)
        )
        s_tokens, ns_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens, ns_tokens, mask, history_gate, score_biases,
        )

        if self.readout == "mean":
            bottom_output = ns_tokens.mean(dim=1)
        else:
            bottom_output = ns_tokens.flatten(start_dim=1)
        bottom_output = torch.cat(
            [bottom_output, self.masked_avg_pooling(s_tokens, mask)], dim=-1
        )
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        return {
            "{}_pred".format(labels[i]): self.output_activation[i](v)
            for i, v in enumerate(tower_output)
        }


def _lca_forward_with_score_bias(self, s_tokens, ns_tokens, mask=None,
                                 score_bias=None):
    """``LightweightCrossAttention.forward`` + optional key-side bias."""
    B, Ls, D = s_tokens.shape
    _, Lns, _ = ns_tokens.shape

    if mask is not None:
        s_tokens = s_tokens * mask.unsqueeze(-1).float()

    k_s = self.W_k_s(s_tokens)
    v_s = self.W_v_s(s_tokens)
    q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)

    q = q_ns.view(B, Lns, self.num_heads, self.head_dim).transpose(1, 2)
    k = k_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
    v = v_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)

    output, _ = self.dot_attention(
        q, k, v, scale=self.head_dim ** 0.5, mask=score_bias,
    )
    output = output.transpose(1, 2).contiguous().view(B, Lns, D)
    # ``W_o`` is a shared projection after the family-wide alignment.
    return self.W_o(output)
