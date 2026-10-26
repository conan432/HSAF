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
from .HyFormer import HyFormer
from .OneTrans import OneTrans
from .RankMixer import RankMixer
from unirank.pytorch.layers.attentions.dot_product_attention import (
    padding_mask_to_sdpa,
)


# Effective-history-length bucket edges (identical to the reference model).
from .EviTime_gatecalib_use_time import (  # noqa: F401  (主线核心，本文件全部继承它)
    EviTimeCoreMixin,
    EVIDENCE_LEN_EDGES, NUM_EVIDENCE_LEN_BUCKETS,
    DEFAULT_EVIDENCE_TIME_BUCKETS, DEFAULT_EVIDENCE_TIME_BASE, DEFAULT_EVIDENCE_DIMS,
    _normalise_evidence_dims, _evidence_time_buckets, _resolve_time_unit,
    _read_evitime_inputs,
    _lca_forward_with_score_bias,
    _mixed_hetero_attention_with_score_bias, _est_blocks_forward, _attach_mixing_gate, _mixer_forward_with_mixing_gate,
)


def _hetero_query_tokenizer_forward_gated(self, feature_embeddings,
                                          sequence_emb, mask=None,
                                          history_gate=None, score_bias=None):
    """Ablation-only history reader: main line read + optional readout gate / FiLM.

    This is the *superset* of the main line's ``_hetero_query_tokenizer_forward``
    that the ablation variants need:

    * ``history_gate`` may be a request-level scalar gate ``(B, 1, 1)`` that
      scales the ``2N`` history tokens once (the "gated history read" mode); or
      the packed FiLM tensor ``(B, 1, gamma|beta)`` of Plan G1e;
    * ``num_real_token > 0`` keeps HeMix's long/short-term split available for
      the ablation matrix.

    It lives here, not in the main line module, so that
    ``EviTime_gatecalib_use_time.py`` only contains what the main line runs.
    """
    if feature_embeddings.dim() > 2:
        feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
    B = feature_embeddings.size(0)
    ns_tokens = self.token_mlps(feature_embeddings)
    film = None
    if (history_gate is not None and torch.is_tensor(history_gate)
            and history_gate.dim() == 3 and history_gate.size(-1) != 1):
        # Plan G1e: packed FiLM tensor (B, 1, gamma|beta) from _compute_evidence.
        film = history_gate
        gate = 1.0
    else:
        gate = 1.0 if history_gate is None else history_gate

    if self.num_real_token > 0:
        real_len = self.real_seq_len
        real_sequence_emb = sequence_emb[:, -real_len:, :]
        glocal_sequence_emb = sequence_emb[:, :-real_len, :]
        if mask is not None:
            real_mask = mask[:, -real_len:]
            glocal_mask = mask[:, :-real_len]
            real_sequence_emb = real_sequence_emb * real_mask.unsqueeze(-1).float()
            glocal_sequence_emb = glocal_sequence_emb * glocal_mask.unsqueeze(-1).float()

        ns_tokens_glocal = ns_tokens[:, :self.num_glocal_token, :]
        ns_tokens_real = ns_tokens[:, self.num_glocal_token:, :]
        real_queries = torch.cat(
            [ns_tokens_real, self.cls_tokens_real.expand(B, -1, -1)], dim=1)
        glocal_queries = torch.cat(
            [ns_tokens_glocal, self.cls_tokens_glocal.expand(B, -1, -1)], dim=1)

        score_bias_real = score_bias_glocal = None
        if score_bias is not None:
            score_bias_real = score_bias[..., -real_len:]
            score_bias_glocal = score_bias[..., :-real_len]

        tokens_real = self.mha_real(
            s_tokens=real_sequence_emb,
            ns_tokens=real_queries,
            score_bias=score_bias_real,
        )
        tokens_glocal = self.mha_glocal(
            s_tokens=glocal_sequence_emb,
            ns_tokens=glocal_queries,
            score_bias=score_bias_glocal,
        )
        if film is not None:
            gamma = film[..., :film.size(-1) // 2]
            beta = film[..., film.size(-1) // 2:]
            tokens_real = gamma * tokens_real + beta
            tokens_glocal = gamma * tokens_glocal + beta
        else:
            tokens_real = gate * tokens_real
            tokens_glocal = gate * tokens_glocal
        history_tokens = torch.cat([tokens_real, tokens_glocal], dim=1)
    else:
        if mask is not None:
            sequence_emb = sequence_emb * mask.unsqueeze(-1).float()
        queries = torch.cat(
            [ns_tokens, self.cls_tokens.expand(B, -1, -1)], dim=1)
        history_tokens = self.mha_full(
            s_tokens=sequence_emb,
            ns_tokens=queries,
            score_bias=score_bias,
        )
        if film is not None:
            gamma = film[..., :film.size(-1) // 2]
            beta = film[..., film.size(-1) // 2:]
            history_tokens = gamma * history_tokens + beta
        else:
            history_tokens = gate * history_tokens

    return torch.cat([history_tokens, ns_tokens], dim=1)


class EviTimeMixin(EviTimeCoreMixin):
    """Shared evidence modules, feature computation and readout hooks."""

    def _init_evitime(self,
                      use_evidence_token=True,
                      use_history_gate=True,
                      use_logit_calibration=True,
                      num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                      evidence_len_edges=None,
                      num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                      evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                      time_unit=None,
                      evidence_dims=DEFAULT_EVIDENCE_DIMS,
                      shuffle_evidence=False,
                      use_temporal_bias=False,
                      use_position_bias=False,
                      center_history_gate=False,
                      center_calibration=False,
                      center_evidence=False,
                      gate_spread_weight=0.0,
                      gate_spread_target=0.15,
                      centered_gate_spread=False,
                      calib_task_specific=False,
                     score_bias_num_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                      score_bias_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                      score_bias_num_layers=None,
                      score_bias_max_len=100,
                      has_history_self_attention=False,
                      use_self_attention_bias=False,
                      history_gate_scale=2.0,
                      use_self_value_gain=False,
                      gate_form="mul",
                      flow_mode="readout",
                      per_layer_gate=False,
                      gate_in_mixing=False,
                      aux_loss_mode="none",
                      aux_loss_weight=0.0,
                      center_mode="batch",
                      center_ema_momentum=0.99,
                      center_ema_warmup_batches=20,
                      evidence_token_zero_init=False,
                      gate_neutral_init=False,
                      calib_neutral_init=False,
                      baseline_teacher_ckpt=None,
                      shift_penalty_weight=0.0,
                      shift_penalty_clamp=1.0e-6,
                      evidence_residual_bottleneck=False,
                      evidence_bottleneck_dim=64,
                      residual_center_weight=0.0,
                      residual_center_mode="ema_penalty",
                      residual_center_ema_momentum=0.99,
                      residual_center_warmup_batches=20,
                      bucket_center_residual=False,
                      residual_center_global_weight=0.0,
                      residual_center_bucket_weight=0.0,
                      residual_bucket_edges=(20, 80),
                      evidence_token_mlp=False,
                      evidence_token_mlp_bottleneck=0,
                      evidence_token_mlp_joint=False,
                      evidence_token_mlp_hybrid=False,
                      evidence_token_film=False,
                      history_film_gate=False,
                      state_channel_weights=False,
                      evidence_window_max=0):
        dim = self.token_dim
        self.use_evidence_token = bool(use_evidence_token)
        self.use_history_gate = bool(use_history_gate)
        # Gate range: ``g = history_gate_scale * sigmoid(W e)``.  With a
        # zero-initialised head this gives ``g = history_gate_scale / 2`` at
        # step 0, so ``2.0`` is the choice that makes the neutral point exactly
        # ``g = 1`` (no re-scaling of the history read) while still allowing the
        # model to go both below and above 1.  It is a *scale* hyper-parameter,
        # not a convention: larger values widen the reachable range.
        self.history_gate_scale = float(history_gate_scale)
        self.use_logit_calibration = bool(use_logit_calibration)
        # ------------------------------------------------------------------
        # Guidance form and where the guidance is applied.
        #   ``gate_form``:
        #     "mul"        g = 2*sigmoid(W e) in [0, 2]  (current default)
        #     "none"       no multiplicative gate at all (guidance = token
        #                  evidence + logit calibration only)
        #     "scaleshift" delta <- (1 + s(e)) * delta + b(e), with s, b both
        #                  zero-initialised: an affine guidance instead of a
        #                  gate, so it can also *add* rather than only scale.
        #   ``flow_mode`` (mixer family only):
        #     "readout"  current behaviour: one extra gated history -> NS path
        #                per layer; the layer's own token mixing is untouched.
        #     "rows"     the gate scales the history rows at the *input of every
        #                layer's token mixing*, so every flow of history
        #                information (history -> NS *and* history -> history)
        #                goes through the gate.  No extra FLOPs.
        #     "exact"    two mixing passes per layer; the gate multiplies only
        #                the history-attributable part of the layer update
        #                (``mix(x) - mix(x with history rows zeroed)``), which
        #                covers all flows exactly at +1 low-rank mixing pass.
        # ------------------------------------------------------------------
        self.gate_form = str(gate_form).strip().lower()
        self.flow_mode = str(flow_mode).strip().lower()
        # ------------------------------------------------------------------
        # M-A ("gate inside the token mixing") and its per-layer variant.
        #   ``gate_in_mixing``: instead of adding a parallel history->NS
        #     attention path (``HistoryToNonSeqGate``) or scaling rows at the
        #     call site, the gate is handed to ``LowRankTokenMixing`` itself and
        #     scales the **history rows of that layer's mixing input**:
        #         x_in = [ g * x_hist , x_ns ] -> LowRankTokenMixing(x_in)
        #     so one ``g`` modulates every downstream flow of history, the
        #     residual stream keeps the ungated representation, and the cost is
        #     one elementwise multiply (zero extra FLOPs).
        #   ``per_layer_gate``: give every interaction layer its **own** gate MLP
        #     (evidence -> scalar) instead of sharing one request-level gate.
        self.gate_in_mixing = bool(gate_in_mixing)
        self.per_layer_gate = bool(per_layer_gate)
        self.aux_loss_mode = str(aux_loss_mode).strip().lower()
        self.aux_loss_weight = float(aux_loss_weight)
        if self.gate_form not in ("mul", "none", "scaleshift"):
            raise ValueError("gate_form must be 'mul', 'none' or 'scaleshift'")
        if self.gate_form == "none":
            # No multiplicative guidance anywhere: the evidence still reaches
            # the model through the token channel and the logit calibration,
            # but nothing scales a history-derived tensor.  Keeping
            # ``use_history_gate`` true here would silently attach an *ungated*
            # history -> NS path, which is not "no gate".
            self.use_history_gate = False
        # ``evidence_axis="all"``: every non-sequence token receives the vector.
        self.evidence_axis = "all"
        # ------------------------------------------------------------------
        # History-length bucketing.  Two ways to configure it:
        #   * ``evidence_len_edges``: explicit **upper-bound** edges produced by
        #     ``observation_experiment/make_len_buckets.py`` from the *training*
        #     split (equal-frequency over the non-saturated lengths, empty
        #     history owning slot 0, and one bucket for ``len == max_len`` which
        #     cannot be split further).  The number of buckets is then
        #     ``len(edges)`` and overwrites ``num_evidence_len_buckets``.
        #   * otherwise the legacy fixed edges (``EVIDENCE_LEN_EDGES``) with
        #     ``num_evidence_len_buckets`` rows are used, so old checkpoints and
        #     configs keep working unchanged.
        # ------------------------------------------------------------------
        if evidence_len_edges is not None:
            edges = [int(e) for e in evidence_len_edges]
            assert edges and edges[0] == 0, "length edges must start with 0 (empty history)"
            assert all(a < b for a, b in zip(edges, edges[1:])), \
                "length edges must be strictly increasing"
            self.evidence_len_edges = tuple(edges)
            self.num_evidence_len_buckets = len(edges)
        else:
            self.evidence_len_edges = tuple(EVIDENCE_LEN_EDGES)
            self.num_evidence_len_buckets = int(num_evidence_len_buckets)
        self.register_buffer(
            "_len_edges_tensor",
            torch.tensor(self.evidence_len_edges, dtype=torch.float64),
            persistent=False,
        )
        self.evidence_num_time_buckets = int(num_evidence_time_buckets)
        self.evidence_time_base = float(evidence_time_base)
        self.evidence_time_unit = _resolve_time_unit(self, time_unit)
        # Dimension ablation: which of (length, staleness, span) actually feed
        # the evidence vector.  Inactive dimensions get **no embedding at all**
        # (as opposed to an embedding that is added with weight 0): an unused
        # parameter would be a DDP-visible dead weight and, more importantly,
        # it keeps the ablation honest - the model literally cannot see the
        # dropped dimension.
        self.evidence_dims = _normalise_evidence_dims(evidence_dims)
        # Control flag: permute the evidence vector across the batch.  A model
        # that still improves with shuffled evidence is not actually using it.
        self.shuffle_evidence = bool(shuffle_evidence)
        # ------------------------------------------------------------------
        # temporal / position score bias (same design as the reference
        # ``TetraAxisNet_dual_memory_evidence`` implementation):
        # two **per-layer, zero-initialised** key-side bias tables added to the
        # attention logits over history keys.
        #   * temporal: log-bucketed gap between the target timestamp and the
        #     timestamp of the key (``floor(log1p(gap_s)/log(base))``);
        #   * position: distance of the key from the *end* of the history.
        # Zero-init means switching them on is a no-op at step 0, so the variant
        # starts exactly where the gate/calibration-only version starts.
        # ------------------------------------------------------------------
        self.use_temporal_bias = bool(use_temporal_bias)
        self.use_position_bias = bool(use_position_bias)
        # ------------------------------------------------------------------
        # De-biasing switches (see ``observation_experiment/evidence_
        # optimization.md``): the measured optimum of the two learnable
        # channels is dominated by a *global* coefficient (an average gate of
        # 0.25-0.81 that shrinks the whole history read, plus a matching
        # negative calibration offset).  That makes EviTime raise every score
        # instead of re-ordering users by how much history backs them.
        #   * ``center_history_gate``: g <- g / mean_batch(g)   (only relative)
        #   * ``center_calibration``:  c <- c - mean_batch(c)   (per task)
        #   * ``center_evidence``:     remove the global component of the
        #     evidence vector itself, so both channels can only see "how does
        #     this sample's history compare with the batch".
        # All three are batch-mean subtractions with a **detached** mean (the
        # mean itself must not be pushed around by gradients).
        # ------------------------------------------------------------------
        self.center_history_gate = bool(center_history_gate)
        self.center_calibration = bool(center_calibration)
        self.center_evidence = bool(center_evidence)
        # ------------------------------------------------------------------
        # Centering mode.  Batch centering guarantees zero batch-mean shift,
        # but makes inference depend on batch composition.  ``ema`` keeps a
        # synchronized running mean during training, then freezes it for eval;
        # this is the deployment-safe version of the same idea.
        # ------------------------------------------------------------------
        self.center_mode = str(center_mode).strip().lower()
        if self.center_mode not in ("batch", "ema"):
            raise ValueError("center_mode must be 'batch' or 'ema'")
        self.center_ema_momentum = float(center_ema_momentum)
        self.center_ema_warmup_batches = int(center_ema_warmup_batches)
        self.register_buffer("_center_step", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        if self.center_mode == "ema":
            self.register_buffer("_evidence_ema_mean", torch.zeros(dim))
            self.register_buffer("_gate_ema_mean", torch.ones(1))
            self.register_buffer("_calibration_ema_mean",
                                 torch.zeros(self.num_tasks))
        # Optional strict no-op initialisation for anchored fine-tuning.
        self.evidence_token_zero_init = bool(evidence_token_zero_init)
        if self.evidence_token_zero_init and self.use_evidence_token:
            self.evidence_token_proj = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.evidence_token_proj.weight)
        # Frozen-teacher setup for Plan B.  Keep the teacher outside Module
        # registries: it must not enter state_dict, optimizer, FLOPs or DDP.
        self.baseline_teacher_ckpt = baseline_teacher_ckpt
        self.shift_penalty_weight = float(shift_penalty_weight)
        self.shift_penalty_clamp = float(shift_penalty_clamp)
        self._baseline_teacher_init_kwargs = None
        self._baseline_teacher = None
        self._last_variant_logits = None
        self._last_baseline_logits = None
        # ------------------------------------------------------------------
        # "Train the two heads properly" auxiliary objectives.
        #
        # The decomposition in ``observation_experiment/evidence_optimization.md``
        # shows that the two learnable channels are *not* learning what they are
        # supposed to: the gate collapses to a near-constant 0.25-0.81 scale
        # (it never decides per sample whether to *amplify* or *shrink* the
        # history read) and the calibration head collapses to a per-task
        # constant (it never applies a task- and evidence-specific correction).
        #
        # Everything below is **label-free** - no positive rate, no bucket
        # statistics, no labels of any kind enter these objectives:
        #
        # * ``gate_spread_weight`` (weight): a hinge penalty, added in
        #   ``add_loss`` (which the framework calls on the model itself, so the
        #   training loop is untouched), that keeps the per-sample spread of
        #   the gate above ``gate_spread_target``.  Without it the gate is free
        #   to collapse onto the constant solution we measured (0.25-0.81).
        # * ``calib_task_specific``: give every task its **own** calibration
        #   MLP instead of a shared trunk with a per-task last layer, so the
        #   calibration is genuinely task-specific rather than four outputs of
        #   one shared function.
        # ------------------------------------------------------------------
        self.gate_spread_weight = float(gate_spread_weight)
        self.gate_spread_target = float(gate_spread_target)
        # Plan D: compute spread on g / stopgrad(mean(g)).  Default remains off
        # so all legacy gatecalib variants keep their exact previous objective.
        self.centered_gate_spread = bool(centered_gate_spread)
        self.calib_task_specific = bool(calib_task_specific)
        self._last_history_gate = None
        self.score_bias_num_time_buckets = int(score_bias_num_time_buckets)
        self.score_bias_time_base = float(score_bias_time_base)
        self.score_bias_max_len = int(score_bias_max_len)
        default_layers = int(
            getattr(self, "num_layers", 0)
            or getattr(getattr(self, "unified_interaction_layers", None), "num_layers", 0)
            or 1
        )
        self.score_bias_num_layers = int(score_bias_num_layers or default_layers)
        # Cheap causal-flow guidance for families whose history rows attend to
        # each other (OneTrans: ``is_causal=True`` over the sequence rows).
        # A per-pair (L x L) logit bias forces SDPA onto a slow, mask-based path
        # (measured 4.58 s/it vs 0.18 s/it).  Instead we modulate the **values**
        # of each history key by a per-key, per-layer gain derived from that
        # key's time/position bucket:
        #     gain_{l,j} = 1 + tanh(g_t[l, t_bucket_j] + g_p[l, pos_j])
        # with both tables zero-initialised (gain == 1 at step 0).  It applies
        # to every flow that reads history (self *and* read), costs O(L) and adds
        # no pairwise tensor.
        # ------------------------------------------------------------------
        self.use_self_value_gain = bool(use_self_value_gain)
        if self.use_self_value_gain:
            self.self_gain_temporal = nn.Parameter(torch.zeros(
                self.score_bias_num_layers, self.evidence_num_time_buckets))
            self.self_gain_position = nn.Parameter(torch.zeros(
                self.score_bias_num_layers, self.score_bias_max_len))
        # ------------------------------------------------------------------
        self.temporal_bias = (
            nn.Parameter(torch.zeros(self.score_bias_num_layers,
                                     self.score_bias_num_time_buckets))
            if self.use_temporal_bias else None
        )
        self.position_bias = (
            nn.Parameter(torch.zeros(self.score_bias_num_layers, self.score_bias_max_len))
            if self.use_position_bias else None
        )
        # Self-attention over history queries needs a **different** bias from
        # cross-attention: HSTU's relative attention bias is defined on the
        # relative quantity between the *query* item and the *key* item, while
        # a cross-attention row (a non-sequence / target query) has no timestamp
        # of its own and must fall back to "key relative to the target".
        # Hence a separate parameter table for the self-attention sites.
        # The pairwise self-attention bias is *implemented and verified*, but it
        # is **off by default**: an explicit (learnable) attention mask forces
        # SDPA onto the math kernel, which measured 5.5x slower at bs=64 and
        # ~100x slower at bs=2048 on the HyFormer self-attention (see the doc
        # §6.8/§6.9).  Cross-attention bias needs only a (B,1,1,L) mask and is
        # cheap, so that is the default.
        self.has_history_self_attention = (
            bool(has_history_self_attention) and bool(use_self_attention_bias)
        )
        self.temporal_bias_self = (
            nn.Parameter(torch.zeros(self.score_bias_num_layers,
                                     self.score_bias_num_time_buckets))
            if (self.use_temporal_bias and self.has_history_self_attention) else None
        )
        self.position_bias_self = (
            # **two-sided** table for self-attention: relative positions range
            # over ``[-(max_len-1), +(max_len-1)]`` because some of these
            # self-attention sites are *not* causal (e.g. HyFormer's
            # SequenceRepresentationLayer attends in both directions), so a
            # one-sided "distance from the end" table would be wrong there.
            nn.Parameter(torch.zeros(self.score_bias_num_layers,
                                     2 * self.score_bias_max_len - 1))
            if (self.use_position_bias and self.has_history_self_attention) else None
        )

        active_embeds = []
        if "len" in self.evidence_dims:
            self.evidence_len_embed = nn.Embedding(self.num_evidence_len_buckets, dim)
            active_embeds.append(self.evidence_len_embed)
        if "staleness" in self.evidence_dims:
            self.evidence_staleness_embed = nn.Embedding(
                self.evidence_num_time_buckets, dim
            )
            active_embeds.append(self.evidence_staleness_embed)
        if "span" in self.evidence_dims:
            self.evidence_span_embed = nn.Embedding(
                self.evidence_num_time_buckets, dim
            )
            active_embeds.append(self.evidence_span_embed)
        if not active_embeds:
            raise ValueError("evidence_dims must keep at least one of len/staleness/span")
        for embed in active_embeds:
            # Small random (not zero): a constant input would stall the gate
            # and calibration MLPs at their first layer.
            nn.init.normal_(embed.weight, std=0.02)

        hidden = max(dim // 4, 16)
        self.evidence_gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        nn.init.normal_(self.evidence_gate[-1].weight, std=1e-2)
        nn.init.zeros_(self.evidence_gate[-1].bias)
        if bool(gate_neutral_init):
            # Exact g=scale*sigmoid(0)=1 at initialization.  Used by anchored
            # fine-tuning so the first forward is identical to the base model.
            nn.init.zeros_(self.evidence_gate[-1].weight)
            nn.init.zeros_(self.evidence_gate[-1].bias)

        if self.per_layer_gate:
            # One gate MLP per interaction layer (same shape/init as the shared
            # head): layer l has its own g_l = scale * sigmoid(W_l e).
            self.evidence_gate_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                for _ in range(int(default_layers))
            ])
            for head in self.evidence_gate_layers:
                nn.init.normal_(head[-1].weight, std=1e-2)
                nn.init.zeros_(head[-1].bias)
                if bool(gate_neutral_init):
                    nn.init.zeros_(head[-1].weight)

        if self.gate_form == "scaleshift":
            # Affine guidance on the history-attributable update: zero-init
            # means the variant starts as the plain model (no gate, no shift).
            self.flow_scale = nn.Sequential(
                nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            self.flow_shift = nn.Sequential(
                nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            for head in (self.flow_scale, self.flow_shift):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        if self.calib_task_specific:
            # One MLP per task: the calibration of task t is a function of the
            # evidence only through task t's own parameters.
            self.evidence_calibration = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                for _ in range(self.num_tasks)
            ])
            for head in self.evidence_calibration:
                nn.init.normal_(head[-1].weight, std=1e-2)
                nn.init.zeros_(head[-1].bias)
                if bool(calib_neutral_init):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
        else:
            self.evidence_calibration = nn.Sequential(
                nn.Linear(dim, hidden), nn.SiLU(),
                nn.Linear(hidden, self.num_tasks),
            )
            nn.init.normal_(self.evidence_calibration[-1].weight, std=1e-2)
            nn.init.zeros_(self.evidence_calibration[-1].bias)
            if bool(calib_neutral_init):
                nn.init.zeros_(self.evidence_calibration[-1].weight)
                nn.init.zeros_(self.evidence_calibration[-1].bias)

        # ------------------------------------------------------------------
        # Anti-shortcut Plan E (evidence_residual_bottleneck): the evidence is
        # barred from the token / gate / calibration channels and acts only
        # through a low-capacity residual head on the final logit,
        #     final_logit_t = tower_logit_t + residual_t,
        #     residual = MLP_bottleneck(LayerNorm(evidence)), zero-init last,
        # so the variant starts behaviourally identical to its tower.  Channel
        # isolation is a config decision; broken combinations (old channels
        # still open) are refused here so an experiment can never run
        # half-isolated.  Global centering keeps E[residual_t] ~ 0 via a soft
        # EMA penalty (default) or a hard stopgrad(EMA) subtraction.
        # ------------------------------------------------------------------
        self.evidence_residual_bottleneck = bool(evidence_residual_bottleneck)
        self.residual_center_mode = str(residual_center_mode).strip().lower()
        self.residual_center_weight = float(residual_center_weight)
        self.residual_center_ema_momentum = float(residual_center_ema_momentum)
        self.residual_center_warmup_batches = int(residual_center_warmup_batches)
        self._last_residual = None
        self._residual_log_once = False
        if self.evidence_residual_bottleneck:
            if self.use_evidence_token or self.use_history_gate \
                    or self.use_logit_calibration:
                raise ValueError(
                    "evidence_residual_bottleneck isolates the evidence from the "
                    "token/gate/calibration channels: set use_evidence_token, "
                    "use_history_gate and use_logit_calibration to false")
            if self.residual_center_mode not in ("ema_penalty", "hard_subtract"):
                raise ValueError(
                    "residual_center_mode must be 'ema_penalty' or 'hard_subtract'")
            bottleneck_dim = max(1, int(evidence_bottleneck_dim))
            self.evidence_residual_norm = nn.LayerNorm(dim)
            self.evidence_residual_head = nn.Sequential(
                nn.Linear(dim, bottleneck_dim),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, self.num_tasks),
            )
            # Zero-init: final logit == tower logit at step 0.
            nn.init.zeros_(self.evidence_residual_head[-1].weight)
            nn.init.zeros_(self.evidence_residual_head[-1].bias)
            self.register_buffer("_residual_ema_mean",
                                 torch.zeros(self.num_tasks))
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanE] active: bottleneck_dim=%d residual_center_weight=%g "
                    "mode=%s zero_init=True ema_buffer=True",
                    bottleneck_dim, self.residual_center_weight,
                    self.residual_center_mode)
        # ------------------------------------------------------------------
        # Anti-shortcut Plan F (bucket_center_residual): keep every EviTime
        # channel, but constrain the residual logit against a frozen same-seed
        # RankMixer teacher (baseline_teacher_ckpt): no global drift, no
        # systematic per-length-bucket shift - only within-bucket re-ranking.
        # Buckets are the half-open effective-length intervals
        # (-inf, e0], (e0, e1], ..., (ek, +inf) over residual_bucket_edges
        # (default three buckets 0-20 / 21-80 / 81-100 at max_len=100).
        # ------------------------------------------------------------------
        self.bucket_center_residual = bool(bucket_center_residual)
        self.residual_center_global_weight = float(residual_center_global_weight)
        self.residual_center_bucket_weight = float(residual_center_bucket_weight)
        if self.bucket_center_residual:
            edges = [int(e) for e in residual_bucket_edges]
            if not edges or any(a >= b for a, b in zip(edges, edges[1:])):
                raise ValueError("residual_bucket_edges must be strictly increasing")
            self.residual_bucket_edges = tuple(edges)
            self.num_residual_buckets = len(edges) + 1
            self.register_buffer(
                "_residual_bucket_edges_tensor",
                torch.tensor(self.residual_bucket_edges, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_bc_global_ema_mean", torch.zeros(self.num_tasks))
            self.register_buffer(
                "_bc_bucket_ema_mean",
                torch.zeros(self.num_tasks, self.num_residual_buckets))
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanF] active: lambda_global=%g lambda_bucket=%g "
                    "bucket_edges=%s teacher=%s",
                    self.residual_center_global_weight,
                    self.residual_center_bucket_weight,
                    list(self.residual_bucket_edges),
                    str(self.baseline_teacher_ckpt))
        if self.evidence_residual_bottleneck or self.bucket_center_residual:
            self.register_buffer(
                "_residual_center_step", torch.zeros(1, dtype=torch.long),
                persistent=False,
            )
        # ------------------------------------------------------------------
        # Plan G1 (evidence_token_mlp): swap channel A from a raw additive
        # embedding sum to one zero-initialised MLP per evidence dimension,
        #     injected = sum_d W2_d SiLU(W1_d e_d),
        # so len/staleness/span learn separate token-space writers.  Channels
        # B/C stay configurable.
        # ------------------------------------------------------------------
        self.evidence_token_mlp = bool(evidence_token_mlp)
        if self.evidence_token_mlp:
            # One zero-initialised head per evidence dimension.  The raw
            # embeddings are no longer summed before the token writer, so
            # len/staleness/span can learn different token-space transforms.
            mid = int(evidence_token_mlp_bottleneck or 0) or dim
            self.evidence_token_mlp_dims = tuple(self.evidence_dims)
            self.evidence_token_mlp_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, mid),
                    nn.SiLU(),
                    nn.Linear(mid, dim),
                )
                for _ in self.evidence_token_mlp_dims
            ])
            for proj in self.evidence_token_mlp_projs:
                nn.init.zeros_(proj[-1].weight)
                nn.init.zeros_(proj[-1].bias)
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanG1] evidence_token_mlp active: split zero-init "
                    "injection heads %s, each dim %d->%d->%d",
                    list(self.evidence_token_mlp_dims), dim, mid, dim)
        # ------------------------------------------------------------------
        # Plan G1b (evidence_token_mlp_joint): the *joint* single-head form
        # used by the 20260926 morning runs: one zero-initialised MLP applied
        # to the summed evidence vector, injected = W2 SiLU(W1 e).
        # Mutually exclusive with the per-dim split head above.
        # ------------------------------------------------------------------
        self.evidence_token_mlp_joint = bool(evidence_token_mlp_joint)
        if self.evidence_token_mlp_joint:
            if self.evidence_token_mlp:
                raise ValueError(
                    "evidence_token_mlp_joint is mutually exclusive with "
                    "evidence_token_mlp (split)")
            mid_j = int(evidence_token_mlp_bottleneck or 0) or dim
            # Attribute name kept as evidence_token_mlp_proj so checkpoints
            # trained by the earlier joint implementation load as-is.
            self.evidence_token_mlp_proj = nn.Sequential(
                nn.Linear(dim, mid_j),
                nn.SiLU(),
                nn.Linear(mid_j, dim),
            )
            nn.init.zeros_(self.evidence_token_mlp_proj[-1].weight)
            nn.init.zeros_(self.evidence_token_mlp_proj[-1].bias)
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanG1b] evidence_token_mlp_joint active: zero-init "
                    "injection head dim %d->%d->%d", dim, mid_j, dim)
        # ------------------------------------------------------------------
        # Plan G1c (evidence_token_mlp_hybrid): keep the raw additive
        # evidence injection (identical to the gatecalib baseline at step 0)
        # and let a zero-initialised MLP learn only a *correction* on top:
        #     injected = e + W2 SiLU(W1 e).
        # This preserves the baseline behaviour that wins on KuaiRand while
        # still giving the evidence a learnable token-space writer.
        # ------------------------------------------------------------------
        self.evidence_token_mlp_hybrid = bool(evidence_token_mlp_hybrid)
        if self.evidence_token_mlp_hybrid:
            if self.evidence_token_mlp or self.evidence_token_mlp_joint:
                raise ValueError(
                    "evidence_token_mlp_hybrid is mutually exclusive with "
                    "evidence_token_mlp / evidence_token_mlp_joint")
            mid_h = int(evidence_token_mlp_bottleneck or 0) or dim
            self.evidence_token_mlp_hybrid_head = nn.Sequential(
                nn.Linear(dim, mid_h),
                nn.SiLU(),
                nn.Linear(mid_h, dim),
            )
            nn.init.zeros_(self.evidence_token_mlp_hybrid_head[-1].weight)
            nn.init.zeros_(self.evidence_token_mlp_hybrid_head[-1].bias)
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanG1c] evidence_token_mlp_hybrid active: raw "
                    "additive + zero-init correction head dim %d->%d->%d",
                    dim, mid_h, dim)
        # ------------------------------------------------------------------
        # Plan G1d (evidence_token_film): FiLM / AdaLN-style injection,
        #     tokens_new = gamma(e) * tokens + beta(e),
        # with gamma = 1 + zero-init head(e) and beta = evidence + zero-init
        # correction head(e).  At step 0 gamma == 1 and beta == evidence, so
        # the injection is exactly the raw additive one used by gatecalib.
        # Mutually exclusive with the mlp / joint / hybrid variants.
        # ------------------------------------------------------------------
        self.evidence_token_film = bool(evidence_token_film)
        if self.evidence_token_film:
            if (self.evidence_token_mlp or self.evidence_token_mlp_joint
                    or self.evidence_token_mlp_hybrid):
                raise ValueError(
                    "evidence_token_film is mutually exclusive with the "
                    "other evidence_token_* injection variants")
            mid_f = int(evidence_token_mlp_bottleneck or 0) or dim
            self.evidence_film_gamma_head = nn.Sequential(
                nn.Linear(dim, mid_f),
                nn.SiLU(),
                nn.Linear(mid_f, dim),
            )
            nn.init.zeros_(self.evidence_film_gamma_head[-1].weight)
            nn.init.zeros_(self.evidence_film_gamma_head[-1].bias)
            self.evidence_film_beta_head = nn.Sequential(
                nn.Linear(dim, mid_f),
                nn.SiLU(),
                nn.Linear(mid_f, dim),
            )
            nn.init.zeros_(self.evidence_film_beta_head[-1].weight)
            nn.init.zeros_(self.evidence_film_beta_head[-1].bias)
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanG1d] evidence_token_film active: gamma=1+zero-init "
                    "head, beta=evidence+zero-init head, dim %d->%d->%d",
                    dim, mid_f, dim)
        # ------------------------------------------------------------------
        # Plan G1e (history_film_gate): replace the scalar history gate with a
        # FiLM modulation on the history readout,
        #     readout_new = gamma(e) * readout + beta(e),
        # gamma = 1 + zero-init head(e), beta = zero-init head(e).  Step 0 is
        # exactly the ungated model.  The packed (B, 1, 2*dim) tensor is
        # carried through the legacy history_gate slot; the scalar-gate paths
        # (spread loss, centering, layer gates) are bypassed because
        # use_history_gate is forced False below.
        # ------------------------------------------------------------------
        self.history_film_gate = bool(history_film_gate)
        if self.history_film_gate:
            if self.use_history_gate:
                raise ValueError(
                    "history_film_gate is mutually exclusive with "
                    "use_history_gate (set use_history_gate: false)")
            hidden_g = max(dim // 4, 16)
            self.history_film_gamma_head = nn.Sequential(
                nn.Linear(dim, hidden_g), nn.SiLU(), nn.Linear(hidden_g, dim))
            nn.init.zeros_(self.history_film_gamma_head[-1].weight)
            nn.init.zeros_(self.history_film_gamma_head[-1].bias)
            self.history_film_beta_head = nn.Sequential(
                nn.Linear(dim, hidden_g), nn.SiLU(), nn.Linear(hidden_g, dim))
            nn.init.zeros_(self.history_film_beta_head[-1].weight)
            nn.init.zeros_(self.history_film_beta_head[-1].bias)
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanG1e] history_film_gate active: readout = gamma(e) * "
                    "readout + beta(e), gamma zero-init 1, beta zero-init 0")
        # ------------------------------------------------------------------
        # Plan H (state_channel_weights): per-channel mixing weights over the
        # raw evidence-part embeddings (len / staleness / span).  h^(ch) =
        # sum_d alpha_d^(ch) * E_d, ch in {injection, gate, calibration};
        # alpha init 1.0 -> step-0 identical to the shared-sum baseline.
        # ------------------------------------------------------------------
        self.state_channel_weights = bool(state_channel_weights)
        if self.state_channel_weights:
            dims = tuple(self.evidence_dims)
            self._scw_dims = dims
            # Plain parameters, NOT a ParameterDict: _evitime_finalize compiles
            # dense child modules and a compiled ParameterDict loses __getitem__.
            for ch in ("inj", "gate", "calib"):
                self.register_parameter(
                    "state_channel_alpha_" + ch,
                    nn.Parameter(torch.ones(len(dims))))
            if self._is_main_process():
                import logging
                logging.info(
                    "[PlanH] state_channel_weights active: per-channel alphas "
                    "over %s, init 1.0", list(dims))
        # ------------------------------------------------------------------
        # Plan G2 (evidence_window_max > 0): compute the evidence features
        # (length / staleness / span buckets) from a *windowed* mask that
        # keeps only the last ``evidence_window_max`` behaviours, so the
        # guidance channels cannot see information carried by the oldest
        # part of the history that the aligned-mixer read path never
        # re-reads (the history read itself stays full-length).
        # ------------------------------------------------------------------
        self.evidence_window_max = int(evidence_window_max or 0)
        if self.evidence_window_max and self._is_main_process():
            import logging
            logging.info(
                "[PlanG2] evidence_window_max active: evidence features see "
                "only the last %d behaviours", self.evidence_window_max)

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

    @torch.no_grad()
    def _sync_center_mean(self, value):
        """Return a rank-local running statistic.

        DDP broadcasts module buffers from rank 0 before later forwards, so
        keeping this as a plain local update avoids a custom collective in the
        model forward.  That is essential with blocked/uneven DDP iterators:
        an all-reduce here can deadlock after ranks exhaust unequal batch
        counts (the framework's Join helper does not cover unknown collectives).
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
    def _update_residual_ema(self, stat_name, batch_mean):
        # Rank-local EMA update for the residual-centering statistics.
        # Mirrors _update_center_stat but with its own step counter (the
        # legacy counter only ticks in center_mode="ema").  Rank-local by
        # design: DDP broadcasts rank-0 buffers before later forwards, so no
        # custom collective ever runs inside the loss (see _sync_center_mean
        # for the uneven-iterator deadlock this avoids).
        stat = getattr(self, stat_name)
        old = stat.detach().clone()
        step = int(self._residual_center_step.item())
        current = batch_mean.detach().to(old.dtype).reshape(old.shape)
        if step <= self.residual_center_warmup_batches:
            new = current
        else:
            momentum = min(max(self.residual_center_ema_momentum, 0.0), 0.999999)
            new = momentum * old + (1.0 - momentum) * current
        stat.copy_(new)
        return new

    def _residual_center_penalty(self, residual_mean, stat_name):
        # Soft EMA penalty on a per-task residual mean: EMA(mean)^2.
        # The recursive EMA cannot carry gradients, so the recursion is
        # expanded exactly one step: the differentiable batch mean enters with
        # weight (1 - momentum) and the frozen running mean anchors the rest.
        # The buffer is refreshed with the detached batch mean, so during the
        # warm-up batches the term reduces to the plain soft penalty mean^2.
        self._update_residual_ema(stat_name, residual_mean)
        momentum = min(max(self.residual_center_ema_momentum, 0.0), 0.999999)
        ema = getattr(self, stat_name).detach().reshape(residual_mean.shape)
        smoothed = momentum * ema + (1.0 - momentum) * residual_mean
        return smoothed.pow(2).sum()

    def _bucket_residual_penalty(self, delta):
        # Plan F bucket term: sum_t sum_b w_b * EMA(mean_{i in b} delta)^2.
        # delta is (B, num_tasks); buckets come from the cached effective
        # history length (_last_length).  w_b = n_b / B (batch population
        # weights).  The EMA buffer is (num_tasks, num_buckets); only the
        # observed columns are refreshed (an empty bucket keeps its previous
        # anchor and contributes 0).
        length = getattr(self, "_last_length", None)
        if length is None or length.numel() == 0:
            return torch.zeros((), device=delta.device)
        edges = self._residual_bucket_edges_tensor.to(length.dtype)
        ids = torch.bucketize(length.reshape(-1), edges)
        total = float(length.numel())
        momentum = min(max(self.residual_center_ema_momentum, 0.0), 0.999999)
        ema = self._bc_bucket_ema_mean
        step = int(self._residual_center_step.item())
        columns, means, weights, anchors = [], [], [], []
        for bucket in range(ema.size(1)):
            mask_b = ids == bucket
            n_b = int(mask_b.sum().item())
            if n_b == 0:
                continue
            columns.append(bucket)
            means.append(delta[mask_b].float().mean(dim=0))
            weights.append(n_b / total)
            anchors.append(ema[:, bucket].detach().clone())
        if not columns:
            return torch.zeros((), device=delta.device)
        # Rank-local EMA refresh of the observed columns (no collectives).
        with torch.no_grad():
            for bucket, mean_b in zip(columns, means):
                if step <= self.residual_center_warmup_batches:
                    new = mean_b.detach().to(ema.dtype)
                else:
                    new = momentum * ema[:, bucket] \
                        + (1.0 - momentum) * mean_b.detach().to(ema.dtype)
                ema[:, bucket] = new
        penalty = None
        for anchor, mean_b, weight_b in zip(anchors, means, weights):
            smoothed = momentum * anchor + (1.0 - momentum) * mean_b.float()
            term = weight_b * smoothed.pow(2).sum()
            penalty = term if penalty is None else penalty + term
        return penalty

    def _build_baseline_teacher(self):
        """Build (once) a frozen same-family teacher for Plan B."""
        if self._baseline_teacher is not None:
            return self._baseline_teacher
        if not self.baseline_teacher_ckpt:
            return None
        from model_zoo.RankMixer import RankMixer
        params = dict(self._baseline_teacher_init_kwargs or {})
        params.setdefault("verbose", 0)
        params["verbose"] = 0
        params["gpu"] = int(getattr(self, "local_rank", 0) or 0)
        params.setdefault("metrics", ["AUC", "logloss"])
        params["metrics"] = params["metrics"]
        params.setdefault("model_root", "/tmp/evitime_teacher")
        params["task"] = ["binary_classification"] * int(params.get("num_tasks", self.num_tasks))
        params.setdefault("loss", ["binary_crossentropy"] * self.num_tasks)
        params["loss"] = ["binary_crossentropy"] * self.num_tasks
        params.setdefault("dense_learning_rate", self.dense_learning_rate)
        # The teacher is never fitted or checkpointed; BaseModel only needs a
        # writable/valid model_root at construction time.
        params["model_root"] = "/tmp/evitime_teacher"
        params["distributed"] = False
        params["rank"] = int(getattr(self, "rank", 0) or 0)
        params["local_rank"] = int(getattr(self, "local_rank", 0) or 0)
        params["world_size"] = int(getattr(self, "world_size", 1) or 1)
        params["enable_torch_compile"] = False
        params["enable_bf16"] = False
        # Do not let teacher construction advance the caller's RNG.
        with torch.random.fork_rng(devices=[] if not torch.cuda.is_available() else [self.device.index or 0]):
            teacher = RankMixer(self.feature_map, model_id="RankMixer", **params)
        state = torch.load(self.baseline_teacher_ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
        teacher.load_state_dict(state, strict=True)
        if self._is_main_process():
            import logging
            logging.info(
                "[PlanF] teacher checkpoint loaded strict=True: %s",
                self.baseline_teacher_ckpt)
        teacher.to(device=self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        # Bypass Module.__setattr__ so the teacher is not saved/optimized/DDP-wrapped.
        object.__setattr__(self, "_baseline_teacher", teacher)
        return teacher

    @torch.no_grad()
    def _baseline_teacher_logits(self, inputs):
        teacher = self._build_baseline_teacher()
        if teacher is None:
            return None
        labels = self.get_active_labels()
        ret = teacher(inputs[:4])
        logits = []
        for name in labels:
            p = ret["{}_pred".format(name)].float().reshape(-1).clamp(
                self.shift_penalty_clamp, 1.0 - self.shift_penalty_clamp)
            logits.append(torch.logit(p))
        return torch.stack(logits, dim=1)

    def _apply_shift_penalty(self, loss):
        """Plan B: permit re-ranking, but penalize the batch mean logit drift."""
        weight = float(getattr(self, "shift_penalty_weight", 0.0))
        if weight <= 0.0 or not self.training:
            return loss
        variant = self._last_variant_logits
        base = self._last_baseline_logits
        if variant is None or base is None:
            return loss
        delta = variant - base
        # With uneven blocked DDP iterators, a custom all-reduce in forward can
        # deadlock.  Penalize each rank-local mean; gradient averaging across
        # ranks still suppresses the shared/global score-shift direction.
        penalty = delta.mean(dim=0).pow(2).sum()
        return loss + weight * penalty

    def _apply_anchored_baseline(self, checkpoint, freeze_base=False):
        """Plan C: load matching base weights and freeze every non-EviTime param."""
        if not checkpoint:
            return
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if self._is_main_process():
            import logging
            logging.info(
                "Anchored baseline loaded: %s (missing=%d, unexpected=%d)",
                checkpoint, len(missing), len(unexpected))
        if not freeze_base:
            return
        trainable_prefixes = (
            "evidence_", "temporal_bias", "position_bias", "self_gain_"
        )
        # Some frozen-attention + trainable-score-bias combinations hit a SDPA
        # backward alignment bug.  Keep the tokenizer/attention upstream graph
        # differentiable, but force its gradients to zero; this preserves the
        # anchored base while avoiding the pathological pruned backward graph.
        keep_graph_prefixes = (
            "item_token_proj.", "unified_tokenizer_layer."
        )
        effective_frozen = 0
        zero_grad_base = 0
        for name, param in self.named_parameters():
            is_evitime = any(name.startswith(prefix) for prefix in trainable_prefixes)
            if is_evitime:
                continue
            if any(name.startswith(prefix) for prefix in keep_graph_prefixes):
                param.requires_grad_(True)
                param.register_hook(lambda grad: torch.zeros_like(grad))
                zero_grad_base += 1
            else:
                param.requires_grad_(False)
                effective_frozen += 1
        if self._is_main_process():
            import logging
            logging.info("Anchored fine-tune: hard-froze %d base parameters; "
                         "kept %d base tensors in the graph with zero-gradient "
                         "hooks; EviTime modules remain trainable.",
                         effective_frozen, zero_grad_base)

    def _evidence_features(self, mask, timestamp_seq, target_timestamp):
        """Bucketised (length, staleness, span) ids for the current batch."""
        window = int(getattr(self, "evidence_window_max", 0) or 0)
        if window > 0 and mask.size(1) > window:
            # Plan G2: the guidance channels may only describe the last
            # ``window`` behaviours; the history read itself stays full-length.
            mask = mask[:, -window:]
            if timestamp_seq is not None and timestamp_seq.numel() > 0:
                timestamp_seq = timestamp_seq[:, -window:]
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
        """Return ``(evidence, history_gate, calibration)``."""
        # Cached for the optional per-task auxiliary loss (no gradient needed).
        self._last_length = mask.sum(dim=1).float().detach()
        # Cached causal-flow guidance (cheap per-key value gains); computed here
        # because this is the only place that sees the timestamps.
        self._last_self_value_gains = self._self_value_gains(
            mask, timestamp_seq, target_timestamp, key_length=mask.size(1)
        )
        len_bucket, staleness_bucket, span_bucket = self._evidence_features(
            mask, timestamp_seq, target_timestamp
        )
        # Keep the uncentered terms for the per-dimension token MLPs.  Other
        # evidence channels (gate/calibration) still consume their sum below.
        evidence_parts = {}
        if "len" in self.evidence_dims:
            evidence_parts["len"] = self.evidence_len_embed(len_bucket)
        if "staleness" in self.evidence_dims:
            evidence_parts["staleness"] = self.evidence_staleness_embed(
                staleness_bucket)
        if "span" in self.evidence_dims:
            evidence_parts["span"] = self.evidence_span_embed(span_bucket)
        self._last_raw_evidence_parts = evidence_parts
        evidence = None
        for term in evidence_parts.values():
            evidence = term if evidence is None else evidence + term
        if getattr(self, "state_channel_weights", False):
            names = list(evidence_parts.keys())
            stacked = torch.stack(
                [evidence_parts[n] for n in names], dim=0)
            alphas = self._scw_alphas
            self._last_channel_h = {
                "inj":   torch.einsum("c,c...->...",
                       alphas["inj"][:len(names)], stacked),
                "gate":  torch.einsum("c,c...->...",
                       alphas["gate"][:len(names)], stacked),
                "calib": torch.einsum("c,c...->...",
                       alphas["calib"][:len(names)], stacked),
            } if alphas is not None else None
        if self.shuffle_evidence and evidence.size(0) > 1:
            evidence = evidence[torch.randperm(evidence.size(0), device=evidence.device)]
        if self.center_evidence:
            evidence = self._center_tensor(evidence, "_evidence_ema_mean")
        history_gate = None
        if getattr(self, "history_film_gate", False):
            gamma = 1.0 + self.history_film_gamma_head(evidence)
            beta = self.history_film_beta_head(evidence)
            history_gate = torch.cat([gamma, beta], dim=-1).unsqueeze(1)
        elif self.use_history_gate and self.gate_form == "mul":
            gate_in = (self._last_channel_h or {}).get("gate", evidence) \
                if getattr(self, "state_channel_weights", False) else evidence
            if self.per_layer_gate:
                # one gate per interaction layer: [L, B, 1]
                gates = []
                for head in self.evidence_gate_layers:
                    g = self.history_gate_scale * torch.sigmoid(head(gate_in))
                    if self.center_history_gate:
                        g = self._center_scalar(g, "_gate_ema_mean")
                    gates.append(g)
                history_gate = torch.stack(gates, dim=0).unsqueeze(-1)  # [L,B,1,1]
            else:
                history_gate = (
                    self.history_gate_scale * torch.sigmoid(self.evidence_gate(gate_in))
                )
                if self.center_history_gate:
                    history_gate = self._center_scalar(history_gate, "_gate_ema_mean")
                history_gate = history_gate.unsqueeze(-1)
        flow_scale = flow_shift = None
        if self.gate_form == "scaleshift":
            # Zero-initialised affine guidance: starts as a no-op.
            flow_scale = self.flow_scale(evidence).unsqueeze(-1)
            flow_shift = self.flow_shift(evidence).unsqueeze(-1)
        calib_in = (self._last_channel_h or {}).get("calib", evidence) \
            if getattr(self, "state_channel_weights", False) else evidence
        calibration = (
            self._calibration_forward(calib_in)
            if self.use_logit_calibration else None
        )
        if calibration is not None and self.center_calibration:
            calibration = self._center_tensor(
                calibration, "_calibration_ema_mean")
        if self.training and self.center_mode == "ema":
            with torch.no_grad():
                self._center_step += 1
        # Cache the current evidence terms for read-only diagnostics.  These
        # attributes are not used by the training objective or model output.
        self._last_evidence = evidence
        self._last_calibration = calibration
        self._last_flow_scale = flow_scale
        self._last_flow_shift = flow_shift
        # ``add_loss`` (called right after this forward pass by the framework)
        # reads the gate from here; it skips the spread penalty when the weight
        # is 0 or the gate is absent.
        self._last_history_gate = history_gate
        return evidence, history_gate, calibration

    def add_loss(self, return_dict, y_true):
        """Task loss + the **label-free** spread penalty that trains the gate.

        Called by the framework (``RankModel.compute_loss``); overriding it on
        the model keeps the change inside this file - the training loop and the
        framework code are untouched.  No label, no positive rate and no bucket
        statistic is used here.
        """
        loss = super().add_loss(return_dict, y_true)
        if not self.training:
            return loss
        gate = self._last_history_gate
        if self.gate_spread_weight > 0 and gate is not None and gate.numel() > 1:
            if gate.dim() >= 4:
                # per-layer gates [L, B, 1, 1]: average the hinge over layers
                penalty = 0.0
                for li in range(gate.size(0)):
                    sg = gate[li].reshape(-1)
                    if self.centered_gate_spread:
                        sg = sg / sg.detach().mean().clamp_min(1e-6)
                    spread = sg.std()
                    floor = torch.as_tensor(self.gate_spread_target,
                                            dtype=spread.dtype, device=spread.device)
                    penalty = penalty + torch.relu(floor - spread)
                loss = loss + self.gate_spread_weight * penalty / gate.size(0)
                gate = None      # already handled
        if self.gate_spread_weight > 0 and gate is not None and gate.numel() > 1:
            spread_gate = gate.reshape(-1)
            if self.centered_gate_spread:
                spread_gate = spread_gate / spread_gate.detach().mean().clamp_min(1e-6)
            spread = spread_gate.std()
            floor = torch.as_tensor(self.gate_spread_target, dtype=spread.dtype,
                                    device=spread.device)
            loss = loss + self.gate_spread_weight * torch.relu(floor - spread)

        # ------------------------------------------------------------------
        # Per-task auxiliary loss.  ``decouple`` penalises the *slope* of each
        # task's logit with respect to the history length inside the batch:
        #     beta_t = Cov_z(len, logit_t) / Var_z(len),   z = standardised len
        #     loss += w * mean_t beta_t^2
        # A model that scores long and short histories the same *given* the
        # evidence cannot lower this term by re-ordering, so it targets exactly
        # the "length -> logit" side effect.
        # ------------------------------------------------------------------
        if self.aux_loss_mode == "decouple" and self.aux_loss_weight > 0:
            length = getattr(self, "_last_length", None)
            labels = self.get_active_labels()
            if length is not None and length.numel() > 1:
                z = length.float().reshape(-1)
                z = (z - z.mean()) / (z.std().clamp_min(1e-6))
                var = z.var(unbiased=False).clamp_min(1e-6)
                penalty = 0.0
                used = 0
                for name in labels:
                    key = "{}_pred".format(name)
                    if key not in return_dict:
                        continue
                    p = return_dict[key].float().reshape(-1).clamp(1e-6, 1 - 1e-6)
                    logit = torch.logit(p)
                    beta = ((z - z.mean()) * (logit - logit.mean())).mean() / var
                    penalty = penalty + beta * beta
                    used += 1
                if used:
                    loss = loss + self.aux_loss_weight * penalty / used
        # ------------------------------------------------------------------
        # Anti-shortcut Plan E: keep the evidence residual globally centered
        # (E[residual_t] ~ 0) with the soft EMA penalty.
        # ------------------------------------------------------------------
        if getattr(self, "evidence_residual_bottleneck", False) \
                and self.residual_center_weight > 0 \
                and self.residual_center_mode == "ema_penalty":
            residual = getattr(self, "_last_residual", None)
            if residual is not None and residual.numel() > 0:
                self._residual_center_step += 1
                penalty = self._residual_center_penalty(
                    residual.float().mean(dim=0), "_residual_ema_mean")
                loss = loss + self.residual_center_weight * penalty
                if not self._residual_log_once and self._is_main_process():
                    import logging
                    logging.info(
                        "[PlanE] residual centering loss active (weight=%g)",
                        self.residual_center_weight)
                    self._residual_log_once = True

        # ------------------------------------------------------------------
        # Anti-shortcut Plan F: constrain the residual logit vs the frozen
        # RankMixer teacher - no global drift, no per-length-bucket drift.
        # ------------------------------------------------------------------
        if getattr(self, "bucket_center_residual", False):
            variant = getattr(self, "_last_variant_logits", None)
            base = getattr(self, "_last_baseline_logits", None)
            if variant is not None and base is not None:
                self._residual_center_step += 1
                clamp = float(getattr(self, "shift_penalty_clamp", 1e-6))
                vlog = torch.logit(
                    variant.float().sigmoid().clamp(clamp, 1.0 - clamp))
                delta = vlog - base.float()
                if self.residual_center_global_weight > 0:
                    penalty = self._residual_center_penalty(
                        delta.mean(dim=0), "_bc_global_ema_mean")
                    loss = loss + self.residual_center_global_weight * penalty
                if self.residual_center_bucket_weight > 0:
                    loss = loss + self.residual_center_bucket_weight * \
                        self._bucket_residual_penalty(delta)
                if not self._residual_log_once and self._is_main_process():
                    import logging
                    logging.info(
                        "[PlanF] residual centering loss active "
                        "(global=%g bucket=%g)",
                        self.residual_center_global_weight,
                        self.residual_center_bucket_weight)
                    self._residual_log_once = True

        loss = self._apply_shift_penalty(loss)
        return loss

    @property
    def _scw_alphas(self):
        if not getattr(self, "state_channel_weights", False):
            return None
        try:
            return {
                "inj": self.state_channel_alpha_inj,
                "gate": self.state_channel_alpha_gate,
                "calib": self.state_channel_alpha_calib,
            }
        except AttributeError:
            return None

    def _token_mlp_evidence(self, evidence):
        """Project each raw evidence dimension separately, then add outputs."""
        parts = getattr(self, "_last_raw_evidence_parts", None)
        dims = getattr(self, "evidence_token_mlp_dims", None)
        if parts and dims:
            projected = [
                proj(parts[name])
                for proj, name in zip(
                    self.evidence_token_mlp_projs, dims)
            ]
            evidence = projected[0]
            for term in projected[1:]:
                evidence = evidence + term
        else:
            evidence = self.evidence_token_mlp_proj(evidence)
        return evidence

    def _inject_evidence(self, ns_tokens, evidence):
        """Channel A: add the evidence vector to every non-sequence token."""
        if not self.use_evidence_token:
            return ns_tokens
        if getattr(self, "state_channel_weights", False):
            ev_inj = (getattr(self, "_last_channel_h", None) or {}).get(
                "inj", evidence)
            return ns_tokens + ev_inj.unsqueeze(1)
        if getattr(self, "evidence_token_film", False):
            gamma = 1.0 + self.evidence_film_gamma_head(evidence)
            evidence = evidence + self.evidence_film_beta_head(evidence)
            return ns_tokens * gamma.unsqueeze(1) + evidence.unsqueeze(1)
        if getattr(self, "evidence_token_mlp", False):
            evidence = self._token_mlp_evidence(evidence)
        elif getattr(self, "evidence_token_mlp_joint", False):
            evidence = self.evidence_token_mlp_proj(evidence)
        elif getattr(self, "evidence_token_mlp_hybrid", False):
            evidence = evidence + self.evidence_token_mlp_hybrid_head(evidence)
        elif getattr(self, "evidence_token_zero_init", False):
            evidence = self.evidence_token_proj(evidence)
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
        if getattr(self, "state_channel_weights", False):
            ev_inj = (getattr(self, "_last_channel_h", None) or {}).get(
                "inj", evidence)
            lo, hi = rows
            return torch.cat([
                tokens[:, :lo, :],
                tokens[:, lo:hi, :] + ev_inj.unsqueeze(1),
                tokens[:, hi:, :],
            ], dim=1)
        if getattr(self, "evidence_token_film", False):
            gamma = 1.0 + self.evidence_film_gamma_head(evidence)
            evidence = evidence + self.evidence_film_beta_head(evidence)
            lo, hi = rows
            return torch.cat([
                tokens[:, :lo, :],
                tokens[:, lo:hi, :] * gamma.unsqueeze(1) + evidence.unsqueeze(1),
                tokens[:, hi:, :],
            ], dim=1)
        if getattr(self, "evidence_token_mlp", False):
            evidence = self._token_mlp_evidence(evidence)
        elif getattr(self, "evidence_token_mlp_joint", False):
            evidence = self.evidence_token_mlp_proj(evidence)
        elif getattr(self, "evidence_token_mlp_hybrid", False):
            evidence = evidence + self.evidence_token_mlp_hybrid_head(evidence)
        elif getattr(self, "evidence_token_zero_init", False):
            evidence = self.evidence_token_proj(evidence)
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

    def _score_biases_self(self, mask, timestamp_seq, target_timestamp,
                           query_length, key_length):
        """Pairwise bias for **history self-attention** rows.

        Follows HSTU's relative-attention-bias semantics: the temporal part uses
        the gap between the *query item* and the *key item* (not the target), and
        the position part uses the relative distance ``j − i`` clipped to
        ``[0, max_len)``.  Both tables are per-layer and zero-initialised.

        Returns ``None`` when the model has no history self-attention or both
        switches are off, and a tuple of ``(B, 1, Lq, Lk)`` tensors otherwise.
        """
        if self.temporal_bias_self is None and self.position_bias_self is None:
            return None
        valid_key = mask[:, :key_length].bool()
        pos_q = torch.arange(query_length, device=mask.device)
        pos_k = torch.arange(key_length, device=mask.device)
        # signed relative position (j - i) -> two-sided table; the +offset makes
        # the index non-negative and the clamp keeps it in range
        offset = self.score_bias_max_len - 1
        relative_position = (
            pos_k.unsqueeze(0) - pos_q.unsqueeze(1) + offset
        ).clamp(0, 2 * self.score_bias_max_len - 2)

        time_buckets = None
        if self.temporal_bias_self is not None and timestamp_seq is not None \
                and timestamp_seq.numel() > 0:
            stamps = timestamp_seq.to(self.device, dtype=torch.float64)[:, :key_length]
            # absolute gap: for causal self-attention this equals the signed gap
            # (keys never follow the query), for the non-causal blocks it keeps
            # the magnitude meaningful in both directions.
            delta = (stamps.unsqueeze(2) - stamps.unsqueeze(1)).abs()
            delta = delta * self._score_bias_time_scale()
            denominator = math.log(max(self.score_bias_time_base, 1.000001))
            time_buckets = torch.floor(torch.log1p(delta) / denominator).long()
            time_buckets = time_buckets.clamp_(0, self.score_bias_num_time_buckets - 1)

        biases = []
        for layer_idx in range(self.score_bias_num_layers):
            bias = None
            if time_buckets is not None:
                bias = self.temporal_bias_self[layer_idx][time_buckets].float()
            if self.position_bias_self is not None:
                term = self.position_bias_self[layer_idx][relative_position].float()
                term = term.unsqueeze(0).expand(mask.size(0), -1, -1)
                bias = term if bias is None else bias + term
            if bias is None:
                biases.append(None)
                continue
            bias = bias * valid_key.unsqueeze(1).float()
            biases.append(bias.view(bias.size(0), 1, bias.size(1), bias.size(2)))
        return tuple(biases)

    def _self_value_gains(self, mask, timestamp_seq, target_timestamp,
                          key_length):
        """Per-key value gains (B, L) per layer, or ``None`` when disabled.

        Cheap stand-in for the per-pair causal logit bias: a key's value is
        scaled by ``1 + tanh(g_time[layer, time_bucket] + g_pos[layer, pos])``,
        so every query that reads that key (the causal self-attention *and* the
        history -> NS read) receives a guided amount of it.  Zero-initialised,
        O(L) cost, no pairwise tensor.
        """
        if not getattr(self, "use_self_value_gain", False):
            return None
        batch_size = mask.size(0)
        pos = torch.arange(key_length, device=mask.device).flip(0)
        pos = pos.clamp_max(self.self_gain_position.size(1) - 1)
        t_buckets = self._score_bias_time_buckets(
            mask, timestamp_seq, target_timestamp)
        gains = []
        for layer_idx in range(self.score_bias_num_layers):
            gain = self.self_gain_position[layer_idx][pos].unsqueeze(0).expand(
                batch_size, -1)
            if t_buckets is not None:
                tb = t_buckets.clamp(0, self.self_gain_temporal.size(1) - 1)
                gain = gain + self.self_gain_temporal[layer_idx][tb]
            gains.append(1.0 + torch.tanh(gain))
        return tuple(gains)


# ---------------------------------------------------------------------------
# temporal / position score bias: the attention-over-history call sites.
#
# Every baseline reads history through *some* attention; the bias is applied as
# a key-side additive mask, i.e. exactly where the reference implementation adds
# it.  Sites and their key sets:
#   * EST      : ``LightweightCrossAttention`` (NS queries, history keys), per layer
#   * OneTrans : mixed MHA over [history, NS] (only the history keys get biased)
#   * HyFormer : history self-attention + query-decoding cross-attention, per layer
#   * HeMix    : the two tokenizer attentions (real / glocal windows), single read
#   * RankMixer: target attention over the history summary, single read
# Passing ``score_bias=None`` reproduces the original module exactly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# temporal / position score bias: the attention-over-history call sites.
#
# Every baseline reads history through *some* attention; the bias is applied as
# a key-side additive mask, i.e. exactly where the reference implementation adds
# it.  Sites and their key sets:
#   * EST      : ``LightweightCrossAttention`` (NS queries, history keys), per layer
#   * OneTrans : mixed MHA over [history, NS] (only the history keys get biased)
#   * HyFormer : history self-attention + query-decoding cross-attention, per layer
#   * HeMix    : the two tokenizer attentions (real / glocal windows), single read
#   * RankMixer: target attention over the history summary, single read
# Passing ``score_bias=None`` reproduces the original module exactly.
# ---------------------------------------------------------------------------


def _sequence_representation_forward_with_score_bias(self, s_tokens, mask=None,
                                                     score_bias=None):
    """``SequenceRepresentationLayer.forward`` (HyFormer) + key-side bias."""
    if mask is not None:
        s_tokens = s_tokens * mask.unsqueeze(-1).float()
    x = s_tokens
    norm_x = self.norm(x)
    B, L, D = norm_x.shape
    q = self.q_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    attn_out, _ = self.dot_attention(
        q, k, v, scale=self.head_dim ** 0.5, mask=score_bias,
    )
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
    attn_out = self.out_proj(attn_out)
    x = attn_out + x
    if mask is not None:
        x = x * mask.unsqueeze(-1).float()
    x = self.ffn(self.ffn_norm(x)) + x
    if mask is not None:
        x = x * mask.unsqueeze(-1).float()
    return x


def _target_attention_with_score_bias(self, target_item, history_sequence,
                                      mask=None, score_bias=None):
    """``MultiHeadTargetAttention.forward`` (RankMixer) + key-side bias."""
    if self.use_qkvo:
        query = self.W_q(target_item)
        key, value = self.W_kv(history_sequence).chunk(2, dim=-1)
    else:
        query, key, value = target_item, history_sequence, history_sequence
    batch_size = query.size(0)
    query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    sdpa_mask, has_history = padding_mask_to_sdpa(
        mask, batch_size=batch_size, length=key.size(-2)
    )
    if score_bias is not None:
        bias = score_bias.to(query.dtype)
        if sdpa_mask is not None:
            sdpa_mask = sdpa_mask.expand(-1, self.num_heads, -1, -1)
            bias = torch.where(sdpa_mask, bias, torch.finfo(query.dtype).min)
        sdpa_mask = bias
    elif sdpa_mask is not None:
        sdpa_mask = sdpa_mask.expand(-1, self.num_heads, -1, -1)

    output, _ = self.dot_attention(query, key, value, scale=self.scale,
                                   mask=sdpa_mask)
    output = output.transpose(1, 2).contiguous().view(
        -1, self.num_heads * self.head_dim)
    if self.use_qkvo:
        output = self.W_o(output)
    if has_history is not None:
        output = output * has_history.to(output.dtype)
    return output


# ---------------------------------------------------------------------------
# S2: gated history -> non-sequence path, one per interaction layer.
# ---------------------------------------------------------------------------

class HistoryToNonSeqGate(nn.Module):
    """Low-rank, per-layer path that carries history into the NS tokens only.

    In a token-mixer the history tokens and the non-sequence tokens are mixed
    **jointly** (``MultiHeadTokenMixing`` flattens all tokens and feeds them to
    one low-rank MLP), so there is no separate "history read" term that could be
    scaled the way ``EST``/``OneTrans``/``HyFormer`` scale theirs.  Scaling the
    history token rows at the mixing input - the obvious workaround - would also
    change the history tokens themselves, which is *not* what the evidence gate
    means: the gate answers "how much history backs this sample", i.e. it should
    control the information flowing **into the non-sequence tokens**.

    This module implements exactly that: for each layer,

        ``delta = W_o( softmax((W_q x_ns)(W_k x_h)^T / sqrt(r)) @ (W_v x_h) )``
        ``x_ns <- x_ns + g * delta``

    where ``x_h`` are the history token rows and ``x_ns`` the non-sequence ones.
    The history tokens are untouched (they are only read), the path is low-rank
    (``r`` = 64 by default: 4 * d * r parameters) and the extra FLOPs are the
    cost of the small attention above - roughly 5% of a mixer layer, versus the
    2x mixing pass that an exact "zero-out the history and take the difference"
    decomposition would need.

    ``W_o`` is initialised small (``N(0, 1e-2^2)``) so the variant starts out
    behaviourally identical to the same model without the S2 path - same
    convention as the evidence gate/calibration heads.
    """

    def __init__(self, dim, rank=64):
        super(HistoryToNonSeqGate, self).__init__()
        self.rank = int(rank)
        self.W_q = nn.Linear(dim, self.rank, bias=False)
        self.W_k = nn.Linear(dim, self.rank, bias=False)
        self.W_v = nn.Linear(dim, self.rank, bias=False)
        self.W_o = nn.Linear(self.rank, dim, bias=False)
        self.scale = float(self.rank) ** 0.5
        for layer in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.W_o.weight, std=1e-2)

    def forward(self, ns_tokens, history_tokens):
        query = self.W_q(ns_tokens)
        key = self.W_k(history_tokens)
        value = self.W_v(history_tokens)
        logits = torch.einsum("bld,bmd->blm", query, key) / self.scale
        weights = torch.softmax(logits, dim=-1)
        return self.W_o(torch.einsum("blm,bmd->bld", weights, value))


def _attach_history_gate(block, dim, num_layers, nonseq_slice, history_slice,
                         rank=64):
    """Attach one ``HistoryToNonSeqGate`` per layer and patch ``forward``."""
    block.history_gate_layers = nn.ModuleList([
        HistoryToNonSeqGate(dim, rank=rank) for _ in range(num_layers)
    ])
    block.nonseq_slice = tuple(nonseq_slice)
    block.history_slice = tuple(history_slice)
    block.forward = types.MethodType(_mixer_forward_with_history_gate, block)
    return block


def _attach_history_flow_gate(block, num_layers, nonseq_slice, history_slice,
                              mode):
    """Route the evidence guidance through the mixer's own token-mixing flow."""
    block.nonseq_slice = tuple(nonseq_slice)
    block.history_slice = tuple(history_slice)
    block.history_flow_mode = str(mode)
    block.forward = types.MethodType(_mixer_forward_with_history_flow, block)
    return block






def _mixer_forward_with_history_flow(self, x, history_gate=None,
                                     flow_scale=None, flow_shift=None):
    """Mixer forward where the evidence guidance touches *every* history flow.

    In a token mixer the history rows are mixed jointly with the other rows, so
    history information reaches the non-sequence tokens (and the later history
    rows) through the layer's own mixing operator.  The readout-only gate of
    ``_mixer_forward_with_history_gate`` is a *parallel* path and therefore
    cannot modulate that dominant flow; these modes can:

      * ``rows``  – the history rows are scaled (by ``g``) at the mixing input
        of every layer, so all downstream flows of history are attenuated;
        zero extra FLOPs.
      * ``exact`` – two mixing passes per layer; the guidance multiplies only
        the history-attributable part of the update
        ``mix(.) - mix(. with history rows zeroed)``, which is the exact
        decomposition of "how much of this layer's update came from history".

    ``gate_form="scaleshift"`` swaps the multiplicative gate for a zero-init
    affine guidance ``(1 + s) * delta + b`` (``b`` only in ``exact`` mode).
    """
    lo, hi = self.nonseq_slice
    h_lo, h_hi = self.history_slice
    mode = getattr(self, "history_flow_mode", "rows")
    g = 1.0 if history_gate is None else history_gate
    for i in range(self.num_layers):
        normed = self.mixer_norms[i](x)
        if mode == "exact":
            full = self.mixer_layers[i](normed)
            zeroed = torch.cat([
                normed[:, :h_lo, :],
                torch.zeros_like(normed[:, h_lo:h_hi, :]),
                normed[:, h_hi:, :],
            ], dim=1)
            nohist = self.mixer_layers[i](zeroed)
            delta = full - nohist
            mixed = nohist + g * delta
            if flow_shift is not None:
                mixed = mixed + flow_shift
        else:  # "rows"
            scale = g
            if flow_scale is not None:
                scale = g * (1.0 + flow_scale) if history_gate is not None \
                    else (1.0 + flow_scale)
            scaled = torch.cat([
                normed[:, :h_lo, :],
                normed[:, h_lo:h_hi, :] * scale,
                normed[:, h_hi:, :],
            ], dim=1)
            mixed = self.mixer_layers[i](scaled)
        x = mixed + x
        x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
    return x


def _mixer_forward_with_history_gate(self, x, history_gate=None):
    """``HeteroMixer`` / ``RankMixerBlock`` forward + one S2 gate per layer.

    Kept structurally identical to the original mixer forward (mixing + FFN
    residuals per layer); the only addition is the gated history -> non-sequence
    term applied to the non-sequence rows *after* each layer, so neither the
    history rows nor the layer's own mixing are modified by the gate.
    """
    lo, hi = self.nonseq_slice
    h_lo, h_hi = self.history_slice
    for i in range(self.num_layers):
        x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
        x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        delta = self.history_gate_layers[i](
            x[:, lo:hi, :], x[:, h_lo:h_hi, :]
        )
        if history_gate is not None:
            delta = history_gate * delta
        x = torch.cat([x[:, :lo, :], x[:, lo:hi, :] + delta, x[:, hi:, :]], dim=1)
    return x


# ---------------------------------------------------------------------------
# EST: the light-weight cross attention output *is* the history read.
# ---------------------------------------------------------------------------



class EST_EviTime(EviTimeMixin, EST):
    """EST + history-evidence conditioning."""

    def __init__(self, feature_map, model_id="EST_EviTime",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 shuffle_evidence=False,
                 use_temporal_bias=False,
                 use_position_bias=False,
                 use_self_attention_bias=False,
                 gate_form="mul",
                 flow_mode="readout",
                 aux_loss_mode="none",
                 aux_loss_weight=0.0,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            history_gate_scale=float(kwargs.get("history_gate_scale", 2.0)),
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            shuffle_evidence=shuffle_evidence,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            use_self_attention_bias=use_self_attention_bias,
            center_history_gate=bool(kwargs.pop("center_history_gate", False)),
            calib_task_specific=bool(kwargs.pop("calib_task_specific", False)),
            gate_spread_weight=float(kwargs.pop("gate_spread_weight", 0.0)),
            gate_spread_target=float(kwargs.pop("gate_spread_target", 0.15)),
            centered_gate_spread=bool(kwargs.pop("centered_gate_spread", False)),
            center_calibration=bool(kwargs.pop("center_calibration", False)),
            center_evidence=bool(kwargs.pop("center_evidence", False)),
            center_mode=str(kwargs.get("center_mode", "batch")),
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            center_ema_warmup_batches=int(kwargs.get("center_ema_warmup_batches", 20)),
            gate_form=kwargs.get("gate_form", "mul"),
            per_layer_gate=bool(kwargs.get("per_layer_gate", False)),
            gate_in_mixing=bool(kwargs.get("gate_in_mixing", False)),
            flow_mode="readout",
            aux_loss_mode=kwargs.get("aux_loss_mode", "none"),
            aux_loss_weight=float(kwargs.get("aux_loss_weight", 0.0)),
            evidence_token_mlp=bool(kwargs.pop("evidence_token_mlp", False)),
            evidence_token_mlp_bottleneck=int(kwargs.pop("evidence_token_mlp_bottleneck", 0)),
            evidence_token_mlp_joint=bool(kwargs.pop("evidence_token_mlp_joint", False)),
            evidence_token_mlp_hybrid=bool(kwargs.pop("evidence_token_mlp_hybrid", False)),
            evidence_token_film=bool(kwargs.pop("evidence_token_film", False)),
            state_channel_weights=bool(kwargs.pop("state_channel_weights", False)),
            score_bias_max_len=int(kwargs.get("max_len", 100)),
        )
        self.unified_interaction_layers.forward = types.MethodType(
            _est_blocks_forward, self.unified_interaction_layers
        )
        for layer in self.unified_interaction_layers.lca_layers:
            layer.forward = types.MethodType(
                _lca_forward_with_score_bias, layer)
        self._evitime_finalize(compile_flag, kwargs)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_evitime_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :].view(batch_size, -1, self.embedding_dim)
        sequence_emb = item_seq_emb[:, 0:-1, :]
        s_tokens = self.item_token_proj(sequence_emb)

        # Same AutoSplit tokenizer input as the aligned EST base.
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat(
            [user_context_emb, target_emb.reshape(batch_size, -1)], dim=-1)
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
            s_tokens,
            ns_tokens,
            mask,
            history_gate,
            score_biases,
        )

        # Same readout switch as the EST base (sequence pool always kept).
        ns_repr = (ns_tokens.mean(dim=1) if getattr(self, "readout", "flatten") == "mean"
                   else ns_tokens.flatten(start_dim=1))
        bottom_output = torch.cat([
            ns_repr,
            self.masked_avg_pooling(s_tokens, mask),
        ], dim=-1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        return {
            "{}_pred".format(labels[index]): self.output_activation[index](value)
            for index, value in enumerate(tower_output)
        }


# ---------------------------------------------------------------------------
# OneTrans: gate only the history-key contribution of the mixed attention.
# ---------------------------------------------------------------------------

def _mixed_mha_forward(self, s_tokens, ps_tokens, ns_tokens,
                       kv_mask=None, q_mask=None, history_gate=None,
                       score_bias=None, self_bias=None, value_gain=None):
    """``MixedMHA.forward`` with an optional per-sample history gate.

    ``q = [q_s, q_ns]`` and ``k = v = [k_s, v_s, k_ns, v_ns]``.  Scaling the
    history *values* by ``g`` scales exactly the part of every query's output
    that came from history keys - the analogue of the reference model's
    "gate the history cross-attention residual".  Because the sequence rows
    (``ps_tokens``) must keep their own (ungated) update, the gated output is
    used only for the non-sequence query rows.
    """
    B, Ls, D = s_tokens.shape
    _, qLs, _ = ps_tokens.shape
    _, Lns, _ = ns_tokens.shape

    if kv_mask is not None:
        s_tokens = s_tokens * kv_mask.unsqueeze(-1).float()

    q_s = self.W_q_s(s_tokens)
    k_s = self.W_k_s(s_tokens)
    v_s = self.W_v_s(s_tokens)

    q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
    k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
    v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)

    q = torch.cat([q_s, q_ns], dim=1)
    k = torch.cat([k_s, k_ns], dim=1)
    v = torch.cat([v_s, v_ns], dim=1)

    L = Ls + Lns
    q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    if value_gain is not None:
        # Per-key gain on the history values (cheap causal-flow guidance): it
        # scales exactly the part of every query's output that came from key j.
        full = value_gain.size(-1)
        if full != Ls:
            value_gain = value_gain[..., max(full - Ls, 0):]
        v = torch.cat([
            v[:, :, :Ls, :] * value_gain[:, None, :, None],
            v[:, :, Ls:, :],
        ], dim=2)

    # ------------------------------------------------------------------
    # Score bias.  Two different kinds live in this single attention call:
    #   * history rows (queries) x history keys  -> **self** attention, biased by
    #     the query-key relative time/position (HSTU RAB semantics);
    #   * NS rows (queries)      x history keys  -> **cross** attention, biased by
    #     "key relative to the target";
    #   * everything x NS keys                   -> no bias.
    # OneTrans prunes the sequence each layer and keeps the *most recent* part,
    # so both biases are sliced to the current window.
    # ------------------------------------------------------------------
    qLs = ps_tokens.size(1)
    key_bias = None
    if score_bias is not None or self_bias is not None:
        # NOTE: OneTrans keeps **all** history positions as queries (the earlier
        # ones are dropped only afterwards via ``output[:, Ls - qLs:Ls]``), so the
        # bias must cover ``Ls + Lns`` query rows, not just the retained ones.
        key_bias = torch.zeros(s_tokens.size(0), 1, Ls + Lns, Ls + Lns,
                               dtype=s_tokens.dtype, device=s_tokens.device)
        if self_bias is not None:
            full = self_bias.size(-1)
            if full != Ls:
                self_bias = self_bias[..., max(full - Ls, 0):, max(full - Ls, 0):]
            key_bias[:, :, :Ls, :Ls] = self_bias
        if score_bias is not None:
            full = score_bias.size(-1)
            if full != Ls:
                score_bias = score_bias[..., max(full - Ls, 0):]
            key_bias[:, :, Ls:, :Ls] = score_bias
    output, _ = self.dot_attention(
        q, k, v, scale=self.head_dim ** 0.5, is_causal=True, mask=key_bias,
    )
    if history_gate is not None:
        gated_v = torch.cat([
            v[:, :, :Ls, :] * history_gate.view(B, 1, 1, 1),
            v[:, :, Ls:, :],
        ], dim=2)
        gated_output, _ = self.dot_attention(
            q, k, gated_v, scale=self.head_dim ** 0.5, is_causal=True,
            mask=key_bias,
        )
        output = torch.cat([
            output[:, :, :Ls, :],
            gated_output[:, :, Ls:, :],
        ], dim=2)

    output = output.transpose(1, 2).contiguous().view(B, L, D)
    output = self.W_o(output)

    ps_out = output[:, Ls - qLs:Ls, :]
    ns_out = output[:, Ls:, :]
    if q_mask is not None:
        ps_out = ps_out * q_mask.unsqueeze(-1).float()
    return ps_out, ns_out


def _onetrans_block_forward(self, s_tokens, ns_tokens, mask=None,
                            history_gate=None, score_biases=None,
                            self_score_biases=None, self_value_gains=None):
    if self_value_gains is None:
        # Cheap causal-flow guidance computed in ``_compute_evidence`` (the
        # forward does not have to thread it through explicitly).
        self_value_gains = getattr(self, "_last_self_value_gains", None)
    if mask is not None:
        s_tokens = s_tokens * mask.unsqueeze(-1).float()
        q_mask = mask
    else:
        q_mask = None

    ps_tokens = s_tokens
    for i in range(self.num_layers):
        num_sequence_tokens = ps_tokens.size(1)
        if self.reduction_ratio >= 1.0:
            sequence_start = 0
        else:
            sequence_start = int(num_sequence_tokens * self.reduction_ratio)
        ps_tokens = ps_tokens[:, sequence_start:, :]
        if q_mask is not None:
            q_mask = q_mask[:, sequence_start:]
        norm_s = self.norm1_layers[i](s_tokens)
        norm_ps = self.norm1_layers[i](ps_tokens)
        norm_ns = self.norm1_layers[i](ns_tokens)

        # ``_compute_evidence`` returns [L, B, 1, 1] when per-layer gates are
        # enabled and [B, 1, 1] when one request-level gate is shared by all
        # layers.  Select the current layer here so OneTrans uses the same
        # layer-specific g_l semantics as the HSAF mainline.
        layer_gate = (history_gate[i]
                      if history_gate is not None and history_gate.dim() == 4
                      else history_gate)
        delta_ps, delta_ns = self.mha_layers[i](
            norm_s, norm_ps, norm_ns,
            kv_mask=mask, q_mask=q_mask, history_gate=layer_gate,
            score_bias=None if score_biases is None else score_biases[i],
            self_bias=None if self_score_biases is None else self_score_biases[i],
            value_gain=None if self_value_gains is None else self_value_gains[i],
        )
        ps_tokens = ps_tokens + delta_ps
        ns_tokens = ns_tokens + delta_ns

        norm_ps = self.norm2_layers[i](ps_tokens)
        norm_ns = self.norm2_layers[i](ns_tokens)
        delta_ps, delta_ns = self.ffn_layers[i](norm_ps, norm_ns, q_mask)
        ps_tokens = ps_tokens + delta_ps
        ns_tokens = ns_tokens + delta_ns

        s_tokens = ps_tokens
        mask = q_mask
    return ns_tokens


class OneTrans_EviTime(EviTimeMixin, OneTrans):
    """OneTrans + history-evidence conditioning."""

    def __init__(self, feature_map, model_id="OneTrans_EviTime",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 shuffle_evidence=False,
                 use_temporal_bias=False,
                 use_position_bias=False,
                 use_self_attention_bias=False,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            history_gate_scale=float(kwargs.get("history_gate_scale", 2.0)),
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            shuffle_evidence=shuffle_evidence,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            use_self_attention_bias=use_self_attention_bias,
            center_history_gate=bool(kwargs.pop("center_history_gate", False)),
            calib_task_specific=bool(kwargs.pop("calib_task_specific", False)),
            gate_spread_weight=float(kwargs.pop("gate_spread_weight", 0.0)),
            gate_spread_target=float(kwargs.pop("gate_spread_target", 0.15)),
            center_calibration=bool(kwargs.pop("center_calibration", False)),
            center_evidence=bool(kwargs.pop("center_evidence", False)),
            center_mode=str(kwargs.get("center_mode", "batch")),
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            center_ema_warmup_batches=int(kwargs.get("center_ema_warmup_batches", 20)),
            gate_form=kwargs.get("gate_form", "mul"),
            per_layer_gate=bool(kwargs.get("per_layer_gate", False)),
            gate_in_mixing=bool(kwargs.get("gate_in_mixing", False)),
            aux_loss_mode=kwargs.get("aux_loss_mode", "none"),
            aux_loss_weight=float(kwargs.get("aux_loss_weight", 0.0)),
            use_self_value_gain=bool(kwargs.get("use_self_value_gain", False)),
            score_bias_max_len=int(kwargs.get("max_len", 100)),
            # this family has history self-attention rows -> HSTU-style relative bias
            has_history_self_attention=True,
        )
        block = self.unified_interaction_layers
        block.forward = types.MethodType(_onetrans_block_forward, block)
        for layer in block.mha_layers:
            layer.forward = types.MethodType(_mixed_mha_forward, layer)
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
        if self.tokenizer_uses_field_input:
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim
            )
        ns_tokens = self.unified_tokenizer_layer(feature_embeddings)

        evidence, history_gate, calibration = self._compute_evidence(
            mask, timestamp_seq, target_timestamp
        )
        ns_tokens = self._inject_evidence(ns_tokens, evidence)

        score_biases = self._score_biases(
            mask, timestamp_seq, target_timestamp, key_length=s_tokens.size(1)
        )
        self_score_biases = self._score_biases_self(
            mask, timestamp_seq, target_timestamp,
            query_length=s_tokens.size(1), key_length=s_tokens.size(1),
        )

        ns_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            ns_tokens,
            mask,
            history_gate,
            score_biases,
            self_score_biases,
        )

        if self.readout == "mean":
            bottom_output = ns_tokens.mean(dim=1)
        else:
            bottom_output = ns_tokens.flatten(start_dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        return {
            "{}_pred".format(labels[index]): self.output_activation[index](value)
            for index, value in enumerate(tower_output)
        }


# ---------------------------------------------------------------------------
# HyFormer: QueryDecoding output is the history read.
# ---------------------------------------------------------------------------

def _query_decoding_forward(self, global_tokens, s_tokens, mask=None,
                            history_gate=None, score_bias=None):
    if mask is not None:
        s_tokens = s_tokens * mask.unsqueeze(-1).float()

    norm_q = self.norm(global_tokens)
    norm_kv = self.norm(s_tokens)

    B, Lq, D = norm_q.shape
    Lk = norm_kv.shape[1]

    # token-specific Q over the query tokens, matching OneTrans' per-token scope
    q_weight = self.q_proj[:Lq]
    q = torch.einsum("bld,ldh->blh", norm_q, q_weight)
    q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

    attn_out, _ = self.dot_attention(q, k, v, scale=self.head_dim ** 0.5,
                                     mask=score_bias)
    if history_gate is not None:
        attn_out = attn_out * history_gate.view(B, 1, 1, 1)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
    attn_out = self.out_proj(attn_out)
    return attn_out + global_tokens


def _hyformer_block_forward(self, s_tokens, global_tokens, ns_tokens,
                            mask=None, history_gate=None, score_biases=None,
                            self_score_biases=None):
    G = global_tokens.size(1)
    for i in range(self.num_layers):
        bias = None if score_biases is None else score_biases[i]
        self_bias = None if self_score_biases is None else self_score_biases[i]
        s_tokens = self.seq_layers[i](s_tokens, mask, score_bias=self_bias)
        decoded_global = self.decode_layers[i](
            global_tokens, s_tokens, mask, history_gate=history_gate,
            score_bias=bias,
        )
        fusion_tokens = torch.cat([decoded_global, ns_tokens], dim=1)
        boosted_tokens = self.boost_layers[i](fusion_tokens)
        global_tokens = boosted_tokens[:, :G, :]
        ns_tokens = boosted_tokens[:, G:, :]
    return s_tokens, global_tokens, ns_tokens


class HyFormer_EviTime(EviTimeMixin, HyFormer):
    """HyFormer + history-evidence conditioning."""

    def __init__(self, feature_map, model_id="HyFormer_EviTime",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 shuffle_evidence=False,
                 use_temporal_bias=False,
                 use_position_bias=False,
                 use_self_attention_bias=False,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            history_gate_scale=float(kwargs.get("history_gate_scale", 2.0)),
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            shuffle_evidence=shuffle_evidence,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            use_self_attention_bias=use_self_attention_bias,
            center_history_gate=bool(kwargs.pop("center_history_gate", False)),
            calib_task_specific=bool(kwargs.pop("calib_task_specific", False)),
            gate_spread_weight=float(kwargs.pop("gate_spread_weight", 0.0)),
            gate_spread_target=float(kwargs.pop("gate_spread_target", 0.15)),
            center_calibration=bool(kwargs.pop("center_calibration", False)),
            center_evidence=bool(kwargs.pop("center_evidence", False)),
            center_mode=str(kwargs.get("center_mode", "batch")),
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            center_ema_warmup_batches=int(kwargs.get("center_ema_warmup_batches", 20)),
            gate_form=kwargs.get("gate_form", "mul"),
            per_layer_gate=bool(kwargs.get("per_layer_gate", False)),
            gate_in_mixing=bool(kwargs.get("gate_in_mixing", False)),
            aux_loss_mode=kwargs.get("aux_loss_mode", "none"),
            aux_loss_weight=float(kwargs.get("aux_loss_weight", 0.0)),
            use_self_value_gain=bool(kwargs.get("use_self_value_gain", False)),
            score_bias_max_len=int(kwargs.get("max_len", 100)),
            # this family has history self-attention rows -> HSTU-style relative bias
            has_history_self_attention=True,
        )
        self.unified_interaction_layers.forward = types.MethodType(
            _hyformer_block_forward, self.unified_interaction_layers
        )
        for layer in self.unified_interaction_layers.decode_layers:
            layer.forward = types.MethodType(_query_decoding_forward, layer)
        for layer in self.unified_interaction_layers.seq_layers:
            layer.forward = types.MethodType(
                _sequence_representation_forward_with_score_bias, layer)
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
        s_tokens = s_tokens * mask.unsqueeze(-1).float()

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        ns_tokens = self.ns_tokenizer(feature_embeddings)

        seq_pool = self.masked_avg_pooling(s_tokens, mask)
        global_info = torch.cat([user_context_emb, target_emb, seq_pool], dim=-1)
        global_tokens = self.query_generator(global_info)

        evidence, history_gate, calibration = self._compute_evidence(
            mask, timestamp_seq, target_timestamp
        )
        ns_tokens = self._inject_evidence(ns_tokens, evidence)

        score_biases = self._score_biases(
            mask, timestamp_seq, target_timestamp, key_length=s_tokens.size(1)
        )
        self_score_biases = self._score_biases_self(
            mask, timestamp_seq, target_timestamp,
            query_length=s_tokens.size(1), key_length=s_tokens.size(1),
        )

        _, global_tokens, final_ns_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            global_tokens,
            ns_tokens,
            mask,
            history_gate,
            score_biases,
            self_score_biases,
        )

        # HyFormer's native readout: mean over the global tokens.
        bottom_output = global_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        return {
            "{}_pred".format(labels[index]): self.output_activation[index](value)
            for index, value in enumerate(tower_output)
        }


# ---------------------------------------------------------------------------
class RankMixer_EviTime(EviTimeMixin, RankMixer):
    """RankMixer + history evidence.

    ``gate_mode="layer"`` (default) keeps the history read ungated and instead
    puts one gated low-rank path per interaction layer; the ``"readout"`` mode
    scales the ``2N`` history tokens produced by the reader once, before the
    mixer layers - the analogue of HeMix's single-read gate.

    Token layout of the aligned mixer family:
    ``[history tokens (2 * num_ns_token) | non-sequence tokens (num_ns_token)]``,
    so the evidence token channel and the layer gate act on the non-sequence
    rows while the history rows keep flowing into the mixer unchanged.
    """

    def __init__(self, feature_map, model_id="RankMixer_EviTime",
                 use_evidence_token=True,
                 use_history_gate=True,
                 use_logit_calibration=True,
                 num_evidence_len_buckets=NUM_EVIDENCE_LEN_BUCKETS,
                 num_evidence_time_buckets=DEFAULT_EVIDENCE_TIME_BUCKETS,
                 evidence_time_base=DEFAULT_EVIDENCE_TIME_BASE,
                 time_unit=None,
                 evidence_dims=DEFAULT_EVIDENCE_DIMS,
                 shuffle_evidence=False,
                 gate_mode="layer",
                 gate_rank=64,
                 use_temporal_bias=False,
                 use_position_bias=False,
                 use_self_attention_bias=False,
                 gate_form="mul",
                 flow_mode="readout",
                 aux_loss_mode="none",
                 aux_loss_weight=0.0,
                 **kwargs):
        compile_flag = bool(kwargs.get("enable_torch_compile", True))
        kwargs["enable_torch_compile"] = False
        # RankMixer's __init__ consumes architecture kwargs by name, so capture
        # them explicitly before super(); the teacher must have identical dims.
        teacher_kwargs = dict(kwargs)
        for key, default in (
            ("embedding_dim", 10),
            ("num_layers", 3),
            ("expansion_factor", 4),
            ("token_dim", 64),
            ("num_ns_token", 5),
            ("num_real_token", 0),
            ("num_heads", 2),
            ("num_mix_heads", 2),
            ("low_rank_dim", 16),
            ("real_seq_len", 20),
            ("tower_hidden_units", [128, 64]),
            ("tower_activations", "ReLU"),
            ("net_dropout", 0),
            ("accumulation_steps", 1),
        ):
            value = kwargs.get(key, default)
            teacher_kwargs.setdefault(key, value)
        super().__init__(feature_map, model_id=model_id, **kwargs)
        self._init_evitime(
            use_evidence_token=use_evidence_token,
            use_history_gate=use_history_gate,
            history_gate_scale=float(kwargs.get("history_gate_scale", 2.0)),
            use_logit_calibration=use_logit_calibration,
            num_evidence_len_buckets=num_evidence_len_buckets,
            evidence_len_edges=kwargs.get("evidence_len_edges", None),
            num_evidence_time_buckets=num_evidence_time_buckets,
            evidence_time_base=evidence_time_base,
            time_unit=time_unit,
            evidence_dims=evidence_dims,
            shuffle_evidence=shuffle_evidence,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            use_self_attention_bias=use_self_attention_bias,
            center_history_gate=bool(kwargs.pop("center_history_gate", False)),
            calib_task_specific=bool(kwargs.pop("calib_task_specific", False)),
            gate_spread_weight=float(kwargs.pop("gate_spread_weight", 0.0)),
            gate_spread_target=float(kwargs.pop("gate_spread_target", 0.15)),
            centered_gate_spread=bool(kwargs.pop("centered_gate_spread", False)),
            center_calibration=bool(kwargs.pop("center_calibration", False)),
            center_evidence=bool(kwargs.pop("center_evidence", False)),
            center_mode=str(kwargs.get("center_mode", "batch")),
            center_ema_momentum=float(kwargs.get("center_ema_momentum", 0.99)),
            center_ema_warmup_batches=int(kwargs.get("center_ema_warmup_batches", 20)),
            gate_form=gate_form,
            per_layer_gate=bool(kwargs.get("per_layer_gate", False)),
            gate_in_mixing=bool(kwargs.get("gate_in_mixing", False)),
            flow_mode=flow_mode,
            aux_loss_mode=aux_loss_mode,
            aux_loss_weight=aux_loss_weight,
            evidence_token_zero_init=bool(kwargs.get("evidence_token_zero_init", False)),
            gate_neutral_init=bool(kwargs.get("gate_neutral_init", False)),
            calib_neutral_init=bool(kwargs.get("calib_neutral_init", False)),
            baseline_teacher_ckpt=kwargs.get("baseline_teacher_ckpt", None),
            shift_penalty_weight=float(kwargs.get("shift_penalty_weight", 0.0)),
            evidence_residual_bottleneck=bool(kwargs.pop("evidence_residual_bottleneck", False)),
            evidence_bottleneck_dim=int(kwargs.pop("evidence_bottleneck_dim", 64)),
            residual_center_weight=float(kwargs.pop("residual_center_weight", 0.0)),
            residual_center_mode=str(kwargs.pop("residual_center_mode", "ema_penalty")),
            residual_center_ema_momentum=float(kwargs.pop("residual_center_ema_momentum", 0.99)),
            residual_center_warmup_batches=int(kwargs.pop("residual_center_warmup_batches", 20)),
            bucket_center_residual=bool(kwargs.pop("bucket_center_residual", False)),
            residual_center_global_weight=float(kwargs.pop("residual_center_global_weight", 0.0)),
            residual_center_bucket_weight=float(kwargs.pop("residual_center_bucket_weight", 0.0)),
            residual_bucket_edges=tuple(kwargs.pop("residual_bucket_edges", (20, 80))),
            evidence_token_mlp=bool(kwargs.pop("evidence_token_mlp", False)),
            evidence_token_mlp_bottleneck=int(kwargs.pop("evidence_token_mlp_bottleneck", 0)),
            evidence_token_mlp_joint=bool(kwargs.pop("evidence_token_mlp_joint", False)),
            evidence_token_mlp_hybrid=bool(kwargs.pop("evidence_token_mlp_hybrid", False)),
            evidence_token_film=bool(kwargs.pop("evidence_token_film", False)),
            history_film_gate=bool(kwargs.pop("history_film_gate", False)),
            state_channel_weights=bool(kwargs.pop("state_channel_weights", False)),
            evidence_window_max=int(kwargs.pop("evidence_window_max", 0)),
            shift_penalty_clamp=float(kwargs.get("shift_penalty_clamp", 1e-6)),
            # RankMixer reads the history once (target attention before the
            # tokenizer), so the score-bias tables hold a single "layer".
            score_bias_num_layers=1,
            score_bias_max_len=int(kwargs.get("max_len", 100)),
        )
        self._baseline_teacher_init_kwargs = teacher_kwargs
        self.gate_mode = str(gate_mode).strip().lower()
        if self.gate_mode not in ("readout", "layer"):
            raise ValueError("gate_mode must be 'readout' or 'layer'")
        # Token layout of the aligned mixer family:
        # [history tokens (2 * num_ns_token) | non-sequence tokens (num_ns_token)]
        self.history_token_slice = (0, 2 * self.num_ns_token)
        self.nonseq_token_slice = (2 * self.num_ns_token, self.num_all_tokens)
        self.history_flow_mode = self.flow_mode
        if self.gate_in_mixing and self.use_history_gate:
            # M-A: the gate lives *inside* the token-mixing operator; no extra
            # module is added (the previous parallel attention path is dropped).
            _attach_mixing_gate(
                self.unified_interaction_layers,
                nonseq_slice=self.nonseq_token_slice,
                history_slice=self.history_token_slice,
            )
        elif self.flow_mode in ("rows", "exact"):
            # Guidance on the mixer's own information flow (all history paths).
            _attach_history_flow_gate(
                self.unified_interaction_layers,
                num_layers=self.unified_interaction_layers.num_layers,
                nonseq_slice=self.nonseq_token_slice,
                history_slice=self.history_token_slice,
                mode=self.flow_mode,
            )
        elif self.gate_mode == "layer" and self.use_history_gate:
            _attach_history_gate(
                self.unified_interaction_layers,
                dim=self.token_dim,
                num_layers=self.unified_interaction_layers.num_layers,
                nonseq_slice=self.nonseq_token_slice,
                history_slice=self.history_token_slice,
                rank=gate_rank,
            )
        # Patched history reader: same read as the base (single window for
        # RankMixer) plus the optional readout gate / key-side score bias.
        self.unified_tokenizer_layer.forward = types.MethodType(
            _hetero_query_tokenizer_forward_gated, self.unified_tokenizer_layer
        )
        for attention in (getattr(self.unified_tokenizer_layer, "mha_full", None),
                          getattr(self.unified_tokenizer_layer, "mha_real", None),
                          getattr(self.unified_tokenizer_layer, "mha_glocal", None)):
            if attention is not None:
                attention.forward = types.MethodType(
                    _mixed_hetero_attention_with_score_bias, attention)
        self._evitime_finalize(compile_flag, kwargs)
        self._apply_anchored_baseline(
            kwargs.get("baseline_load_checkpoint"),
            freeze_base=bool(kwargs.get("freeze_base_parameters", False)),
        )

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

        use_flow = self.flow_mode in ("rows", "exact")
        use_mix_gate = bool(self.gate_in_mixing and self.use_history_gate)
        if (self.gate_mode == "layer" and not getattr(self, "history_film_gate", False)) \
                or use_flow or use_mix_gate:
            # ungated history read; the gate acts once per interaction layer
            unified_tokens = self.unified_tokenizer_layer(
                feature_embeddings, sequence_tokens, mask, score_bias=score_bias
            )
        else:
            # gated history read: the gate scales the 2N history tokens once
            # (``history_gate`` is shaped (B, 1, 1) and broadcasts row-wise).
            unified_tokens = self.unified_tokenizer_layer(
                feature_embeddings, sequence_tokens, mask,
                history_gate=history_gate, score_bias=score_bias,
            )

        unified_tokens = self._inject_evidence_rows(
            unified_tokens, evidence, self.nonseq_token_slice
        )

        if use_mix_gate and history_gate is not None:
            # M-A: the operator itself applies the per-layer gate to its input
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens, history_gate
            )
        elif use_flow:
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens,
                history_gate, self._last_flow_scale, self._last_flow_shift,
            )
        elif self.gate_mode == "layer" and history_gate is not None and not getattr(self, "history_film_gate", False):
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens, history_gate
            )
        else:
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens
            )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        if getattr(self, "evidence_residual_bottleneck", False):
            # Plan E: the evidence touches the logit ONLY through this
            # low-capacity zero-init residual head.
            residual = self.evidence_residual_head(
                self.evidence_residual_norm(evidence))
            self._last_residual = residual
            tower_output = [
                tower_output[i] + residual[:, i:i + 1]
                for i in range(self.num_tasks)
            ]
        tower_output = self._apply_calibration(tower_output, calibration)
        labels = self.get_active_labels()
        if (self.shift_penalty_weight > 0
                or getattr(self, "bucket_center_residual", False)) and self.training:
            self._last_variant_logits = torch.cat(
                [value.float().reshape(-1, 1) for value in tower_output], dim=1)
            self._last_baseline_logits = self._baseline_teacher_logits(inputs)
        return {
            "{}_pred".format(labels[index]): self.output_activation[index](value)
            for index, value in enumerate(tower_output)
        }


class RankMixer_EviTime_layergate(RankMixer_EviTime):
    """``RankMixer_EviTime`` with the gate moved from the readout to every layer."""

    def __init__(self, feature_map, model_id="RankMixer_EviTime_layergate",
                 **kwargs):
        kwargs["gate_mode"] = "layer"
        super().__init__(feature_map, model_id=model_id, **kwargs)
