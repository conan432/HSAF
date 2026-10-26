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
"""HSTU baseline aligned with the unified-architecture family.

``HSTU`` here is the joint (layer-wise) baseline used next to ``OneTrans`` /
``EST`` / the joint version of HSAF.  The file separates two things on
purpose.

**Interface alignment** - everything that the paper's fairness protocol keeps
identical inside one information-flow family:

* one ``AutoSplitTokenizer`` turns the user / context / target fields into
  ``num_ns_token`` non-sequence tokens; the target item is an NS token, so the
  history itself carries no candidate token;
* the history stays the full padded sequence of ``max_len`` tokens and is not
  compressed into a summary;
* ``token_dim`` / layer count / head count / expansion factor / tower / dropout
  / batch size / maximum history length follow the same table as OneTrans and
  EST;
* the readout has the same shape as EST, the joint backbone: mean-pooled NS
  tokens concatenated with the masked average of the history tokens.

**HSTU's own design** - kept, and reported as the baseline's contribution:

* a single self-attention stack over the concatenated ``[history ;
  non-sequence]`` token sequence, i.e. one modular block for the whole input;
* the HSTU block ``Y = f2(Norm(A) . SiLU(U))``; the gating makes a separate
  feed-forward layer unnecessary;
* the semi-local attention mask (causal, plus a local window of ``k1``
  neighbours, plus the last ``k2`` queries attending the whole history);
* the HSTU relative attention bias ``rab_{p,t}``: a relative-position table
  plus a log-bucketed relative-time table for ``(t_j - t_i)``, one table per
  layer and shared across heads, i.e. ``R^{1 x N x N}`` as in the HSTU paper.

A non-sequence query carries the request timestamp, exactly as the candidate
token carries the request time in the original HSTU input, so the temporal part
of ``rab`` expresses how stale every behaviour is with respect to the request
being scored.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedAveragePooling
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer, FieldWiseTokenizer
from unirank.pytorch.models import MultiTaskModel
from .EST import MixedFFN

DEFAULT_TEMPORAL_BIAS_BUCKETS = 64
DEFAULT_TEMPORAL_BIAS_BASE = 2.0


class _BucketLookup(torch.autograd.Function):
    """``table[index]`` with a scatter-add backward that survives duplicates.

    ``torch``'s default backward for an advanced-index lookup
    (``aten::_index_put_impl_`` with ``accumulate=True``) degrades by orders of
    magnitude when the index tensor is highly repetitive, which is exactly the
    case for log-spaced time buckets (most pairs of a long history fall into the
    top bucket).  ``index_add_`` keeps the same semantics at a flat cost.
    """

    @staticmethod
    def forward(ctx, table, index):
        ctx.save_for_backward(index)
        ctx.num_buckets = table.numel()
        return table[index]

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors
        grad_table = grad_output.new_zeros(ctx.num_buckets).index_add_(
            0, index.reshape(-1), grad_output.reshape(-1))
        return grad_table, None


def _resolve_time_unit(feature_map, time_unit=None):
    """Timestamp unit of the dataset, mirroring the EviTime convention."""
    if time_unit:
        return str(time_unit).strip().lower()
    dataset_id = str(getattr(feature_map, "dataset_id", "")).lower()
    if "kuairand" in dataset_id:
        return "ms"
    return "s"


def _time_scale(time_unit):
    """Multiplier that converts the stored timestamp unit into seconds."""
    if time_unit in ("ms", "millisecond", "milliseconds"):
        return 1.0 / 1000.0
    if time_unit in ("us", "microsecond", "microseconds"):
        return 1.0 / 1000000.0
    return 1.0


def _read_timestamps(inputs):
    """History timestamps / target timestamp appended by the collator."""
    timestamp_seq = inputs[4] if len(inputs) > 4 else None
    target_timestamp = inputs[5] if len(inputs) > 5 else None
    return timestamp_seq, target_timestamp


def _as_layer_list(value, num_layers, default):
    """Broadcast a scalar / short list into one entry per layer."""
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        items = [int(v) for v in value]
        if not items:
            items = [int(default[0] if isinstance(default, (list, tuple)) else default)]
        if len(items) >= num_layers:
            return items[:num_layers]
        return items + [items[-1]] * (num_layers - len(items))
    return [int(value)] * num_layers


class HSTU(MultiTaskModel):
    """HSTU baseline with the unified-architecture token interface."""

    def __init__(self,
                 feature_map,
                 model_id="HSTU",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 num_heads=2,
                 token_dim=64,
                 num_ns_token=16,
                 expansion_factor=4,
                 projection_style="legacy",
                 use_mixed_ffn=False,
                 dnn_activations="ReLU",
                 tokenizer_type="Auto",
                 tokenizer_activation="SiLU",
                 readout="mean",
                 use_seq_pool=True,
                 attention_mask="hstu",
                 k1=None,
                 k2=None,
                 truncation_start_layer=None,
                 truncation_ratio=1.0,
                 use_temporal_bias=True,
                 use_position_bias=True,
                 temporal_bias_num_buckets=DEFAULT_TEMPORAL_BIAS_BUCKETS,
                 temporal_bias_base=DEFAULT_TEMPORAL_BIAS_BASE,
                 position_bias_max_distance=None,
                 time_unit=None,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(HSTU, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   **kwargs)

        self.num_tasks = int(kwargs.get("num_tasks", len(self.get_active_labels())))
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps

        readout = str(readout).strip().lower()
        if readout not in ("flatten", "mean"):
            raise ValueError("readout must be 'flatten' or 'mean', got {!r}".format(readout))
        self.readout = readout
        self.use_seq_pool = bool(use_seq_pool)

        attention_mask = str(attention_mask).strip().lower()
        if attention_mask not in ("hstu", "causal"):
            raise ValueError(
                "attention_mask must be 'hstu' or 'causal', got {!r}".format(attention_mask))
        self.attention_mask = attention_mask

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.non_item_dim = 0
        self.num_tokenizer_fields = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim
            self.num_tokenizer_fields += 1

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Same non-sequence token budget as the rest of the family: the target
        # item is an NS token together with the user / context fields.
        self.tokenizer_type = str(tokenizer_type).strip().lower()
        if self.tokenizer_type not in ("auto", "field"):
            raise ValueError("HSTU tokenizer_type must be Auto or Field")
        self.num_ns_token = (self.num_tokenizer_fields if self.tokenizer_type == "field"
                             else int(num_ns_token))
        tokenizer_input_dim = self.non_item_dim + self.item_info_dim
        if self.tokenizer_type == "field":
            if tokenizer_input_dim != self.num_ns_token * embedding_dim:
                raise ValueError("Field tokenizer requires uniform embedding dimensions")
            self.unified_tokenizer_layer = FieldWiseTokenizer(
                input_dim=embedding_dim, token_dim=token_dim,
                num_tokens=self.num_ns_token, activation=tokenizer_activation)
        else:
            self.unified_tokenizer_layer = AutoSplitTokenizer(
                input_dim=tokenizer_input_dim, token_dim=token_dim,
                num_tokens=self.num_ns_token)

        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.masked_avg_pooling = MaskedAveragePooling()

        self.unified_interaction_layers = HSTUInteractionBlocks(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            projection_style=projection_style,
            use_mixed_ffn=use_mixed_ffn,
            num_ns_token=self.num_ns_token,
            dnn_activations=dnn_activations,
            net_dropout=net_dropout,
            attention_mask=self.attention_mask,
            k1=_as_layer_list(k1, num_layers, (32, 32, 16)),
            k2=_as_layer_list(k2, num_layers, (20, 10, 10)),
            truncation_start_layer=truncation_start_layer,
            truncation_ratio=truncation_ratio,
            use_temporal_bias=use_temporal_bias,
            use_position_bias=use_position_bias,
            temporal_bias_num_buckets=temporal_bias_num_buckets,
            temporal_bias_base=temporal_bias_base,
            position_bias_max_distance=(
                int(kwargs.get("max_len", 100))
                if position_bias_max_distance is None else int(position_bias_max_distance)),
            time_scale=_time_scale(_resolve_time_unit(feature_map, time_unit)),
        )

        ns_readout_dim = (token_dim if self.readout == "mean"
                          else token_dim * self.num_ns_token)
        tower_input_dim = ns_readout_dim + (token_dim if self.use_seq_pool else 0)
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=tower_input_dim,
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(self.num_tasks)
        ])
        if isinstance(task, list):
            assert len(task) == self.num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(self.num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_timestamps(inputs)
        batch_size = mask.shape[0]

        # item_dict contains [history_items..., target_item]
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]              # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]          # B x T x item_info_dim
        s_tokens = self.item_token_proj(sequence_emb)    # B x T x token_dim

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        if self.tokenizer_type == "field":
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_ns_token, self.embedding_dim)
        ns_tokens = self.unified_tokenizer_layer(feature_embeddings)   # B x Lns x D

        # Padding stays neutral: the padded history slots enter the stack as
        # zeros (same convention as OneTrans / EST) and are masked out as keys.
        s_tokens = s_tokens * mask.unsqueeze(-1).float()
        tokens = torch.cat([s_tokens, ns_tokens], dim=1)

        tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            tokens,
            mask,
            timestamp_seq,
            target_timestamp
        )

        # Truncation can shorten the history block.  Keep the most recent
        # history tokens and align the readout mask with that shortened block.
        history_len = tokens.size(1) - ns_tokens.size(1)
        s_out = tokens[:, :history_len, :]
        ns_out = tokens[:, history_len:, :]
        seq_mask = mask[:, -history_len:]

        bottom_output = (ns_out.mean(dim=1) if self.readout == "mean"
                         else ns_out.flatten(start_dim=1))
        if self.use_seq_pool:
            bottom_output = torch.cat(
                [bottom_output, self.masked_avg_pooling(s_out, seq_mask)], dim=-1
            )

        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class HSTUInteractionBlocks(nn.Module):
    """Self-attention stack over ``[history ; non-sequence]`` tokens."""

    def __init__(self,
                 token_dim,
                 num_layers,
                 num_heads=2,
                 expansion_factor=4,
                 projection_style="legacy",
                 use_mixed_ffn=False,
                 num_ns_token=None,
                 dnn_activations="ReLU",
                 net_dropout=0.0,
                 attention_mask="hstu",
                 k1=(32, 32, 16),
                 k2=(20, 10, 10),
                 truncation_start_layer=None,
                 truncation_ratio=1.0,
                 use_temporal_bias=True,
                 use_position_bias=True,
                 temporal_bias_num_buckets=DEFAULT_TEMPORAL_BIAS_BUCKETS,
                 temporal_bias_base=DEFAULT_TEMPORAL_BIAS_BASE,
                 position_bias_max_distance=100,
                 time_scale=1.0):
        super(HSTUInteractionBlocks, self).__init__()
        self.num_layers = int(num_layers)
        self.use_mixed_ffn = bool(use_mixed_ffn)
        if self.use_mixed_ffn and (num_ns_token is None or int(num_ns_token) <= 0):
            raise ValueError("num_ns_token is required when use_mixed_ffn=True")
        self.attention_mask = str(attention_mask).strip().lower()
        self.k1 = [int(v) for v in k1]
        self.k2 = [int(v) for v in k2]
        assert len(self.k1) == self.num_layers and len(self.k2) == self.num_layers
        self.truncation_start_layer = (None if truncation_start_layer is None
                                       else int(truncation_start_layer))
        self.truncation_ratio = float(truncation_ratio)

        self.use_temporal_bias = bool(use_temporal_bias)
        self.use_position_bias = bool(use_position_bias)
        self.temporal_bias_num_buckets = int(temporal_bias_num_buckets)
        self.temporal_bias_base = float(temporal_bias_base)
        self.position_bias_max_distance = int(position_bias_max_distance)
        self.time_scale = float(time_scale)

        # ``rab_{p,t}``: one table per layer, shared across heads.  Both tables
        # start at zero, so the bias is a strict no-op at initialisation.
        self.position_bias = (
            nn.Parameter(torch.zeros(self.num_layers, self.position_bias_max_distance + 1))
            if self.use_position_bias else None)
        self.temporal_bias = (
            nn.Parameter(torch.zeros(self.num_layers, self.temporal_bias_num_buckets))
            if self.use_temporal_bias else None)

        self.stu_layers = nn.ModuleList([
            HSTUBlock(
                token_dim=token_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                projection_style=projection_style,
                net_dropout=net_dropout,
            )
            for _ in range(self.num_layers)
        ])
        if self.use_mixed_ffn:
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(token_dim) for _ in range(self.num_layers)
            ])
            self.ffn_layers = nn.ModuleList([
                MixedFFN(token_dim, int(num_ns_token), dnn_activations,
                         expansion_factor=expansion_factor)
                for _ in range(self.num_layers)
            ])
        self._mask_cache = {}

    # -- attention layout ---------------------------------------------------

    def _structural_mask(self, total_len, history_len, k1, k2, device):
        """HSTU semi-local mask on ``[history ; non-sequence]``."""
        cache_key = (int(total_len), int(history_len), int(k1), int(k2),
                     self.attention_mask, device.type, device.index)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        index = torch.arange(total_len, device=device)
        query = index.view(-1, 1)
        key = index.view(1, -1)
        causal = key <= query

        if self.attention_mask == "causal":
            mask = causal
        else:
            is_history_query = query < history_len
            in_local_window = (query - key) < int(k1)
            recent_global_query = query >= (history_len - int(k2))
            history_rule = causal & (in_local_window | recent_global_query)
            # Non-sequence queries are the last positions of the sequence and
            # read the whole prefix (all history plus earlier NS tokens), which
            # is the same layout OneTrans uses for its causal mixed attention.
            mask = torch.where(is_history_query, history_rule, causal)

        self._mask_cache[cache_key] = mask
        return mask

    def _attention_allow(self, history_len, key_valid, layer_index):
        """``(B, S, S)`` boolean mask: structural mask + valid keys + diagonal."""
        total_len = key_valid.size(1)
        structural = self._structural_mask(
            total_len=total_len,
            history_len=history_len,
            k1=self.k1[layer_index],
            k2=self.k2[layer_index],
            device=key_valid.device,
        )
        allow = structural.unsqueeze(0) & key_valid.unsqueeze(1)
        # A query always keeps itself, so an all-padding row can never produce
        # an all ``-inf`` row (its value is zero anyway).
        diagonal = torch.eye(total_len, dtype=torch.bool, device=key_valid.device)
        return allow | diagonal.unsqueeze(0)

    def _time_buckets(self, all_time, key_valid):
        """Log-bucketed relative time ``(B, S, S)``; computed once per layout."""
        if self.temporal_bias is None or all_time is None:
            return None
        gap = (all_time.unsqueeze(2) - all_time.unsqueeze(1)).clamp_min(0.0)
        denominator = math.log(max(self.temporal_bias_base, 1.000001))
        bucket = torch.floor(torch.log1p(gap) / denominator).long()
        bucket = bucket.clamp_(0, self.temporal_bias_num_buckets - 1)
        return bucket * (key_valid.unsqueeze(2) & key_valid.unsqueeze(1)).long()

    def _relative_bias(self, layer_index, time_bucket, key_valid):
        """``rab_{p,t}`` as an additive ``(B, 1, S, S)`` term."""
        total_len = key_valid.size(1)
        bias = None

        if self.position_bias is not None:
            index = torch.arange(total_len, device=key_valid.device)
            distance = (index.view(-1, 1) - index.view(1, -1)).clamp(
                min=0, max=self.position_bias_max_distance)
            bias = self.position_bias[layer_index][distance]

        if time_bucket is not None:
            term = _BucketLookup.apply(self.temporal_bias[layer_index], time_bucket)
            bias = term if bias is None else bias.unsqueeze(0) + term

        if bias is None:
            return None
        if bias.dim() == 2:
            bias = bias.unsqueeze(0).expand(key_valid.size(0), -1, -1)
        return bias.unsqueeze(1)

    # -- sequence layout ----------------------------------------------------

    def _truncate_history(self, tokens, history_len, key_valid, all_time):
        """HSTU's sequence shortening: keep the most recent history tokens."""
        keep = max(1, int(history_len * self.truncation_ratio))
        if keep >= history_len:
            return tokens, history_len, key_valid, all_time
        recent = slice(history_len - keep, history_len)
        tokens = torch.cat([tokens[:, recent, :], tokens[:, history_len:, :]], dim=1)
        key_valid = torch.cat([key_valid[:, recent], key_valid[:, history_len:]], dim=1)
        if all_time is not None:
            all_time = torch.cat([all_time[:, recent], all_time[:, history_len:]], dim=1)
        return tokens, keep, key_valid, all_time

    def _build_time_axis(self, tokens, mask, timestamp_seq, target_timestamp, history_len):
        """Absolute time of every token; NS tokens carry the request time."""
        if not self.use_temporal_bias or timestamp_seq is None or timestamp_seq.numel() == 0:
            return None
        device = tokens.device
        stamps = timestamp_seq[:, :history_len].to(device=device, dtype=torch.float64)
        stamps = stamps * self.time_scale
        ns_len = tokens.size(1) - history_len

        last_stamp = torch.where(
            mask.bool(), stamps, torch.full_like(stamps, float("-inf"))
        ).max(dim=1).values
        last_stamp = torch.where(
            mask.bool().any(dim=1), last_stamp, torch.zeros_like(last_stamp))

        if target_timestamp is not None:
            target = target_timestamp.to(device=device, dtype=torch.float64).view(-1)
            request_time = torch.where(target > 0, target * self.time_scale, last_stamp)
        else:
            request_time = last_stamp
        return torch.cat(
            [stamps, request_time.unsqueeze(1).expand(-1, ns_len)], dim=1)

    def forward(self, tokens, valid_mask, timestamp_seq=None, target_timestamp=None):
        history_len = int(valid_mask.size(1))
        key_valid = valid_mask.bool()
        ns_valid = torch.ones(
            tokens.size(0), tokens.size(1) - history_len,
            dtype=torch.bool, device=tokens.device)
        key_valid = torch.cat([key_valid, ns_valid], dim=1)
        all_time = self._build_time_axis(
            tokens, valid_mask, timestamp_seq, target_timestamp, history_len)
        time_bucket = self._time_buckets(all_time, key_valid)

        for layer_index in range(self.num_layers):
            if (self.truncation_start_layer is not None
                    and layer_index == self.truncation_start_layer
                    and self.truncation_ratio < 1.0):
                tokens, history_len, key_valid, all_time = self._truncate_history(
                    tokens, history_len, key_valid, all_time)
                time_bucket = self._time_buckets(all_time, key_valid)

            allow = self._attention_allow(history_len, key_valid, layer_index)
            bias = self._relative_bias(layer_index, time_bucket, key_valid)
            tokens = self.stu_layers[layer_index](
                tokens, score_mask=allow, score_bias=bias) + tokens
            if self.use_mixed_ffn:
                normalized = self.ffn_norms[layer_index](tokens)
                delta_s, delta_ns = self.ffn_layers[layer_index](
                    normalized[:, :history_len, :],
                    normalized[:, history_len:, :],
                    key_valid[:, :history_len],
                )
                tokens = tokens + torch.cat([delta_s, delta_ns], dim=1)

        return tokens


class HSTUBlock(nn.Module):
    """The HSTU sequential transduction unit ``Y = f2(Norm(A) . SiLU(U))``."""

    def __init__(self, token_dim, num_heads=2, expansion_factor=4,
                 projection_style="legacy", net_dropout=0.0):
        super(HSTUBlock, self).__init__()
        assert token_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.token_dim = token_dim
        self.projection_style = projection_style

        self.pre_norm = nn.LayerNorm(token_dim)
        self.attn_norm = nn.LayerNorm(token_dim)

        if projection_style == "paper":
            # HSTU's f1/f2 are single linear layers; U/Q/K/V each use D.
            self.pre_proj = nn.Linear(token_dim, token_dim * 4)
            self.post_proj = nn.Linear(token_dim, token_dim)
        elif projection_style == "legacy":
            hidden_dim = token_dim * expansion_factor
            self.pre_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim, bias=False),
                nn.SiLU(), nn.Dropout(net_dropout),
                nn.Linear(hidden_dim, token_dim * 4, bias=False),
            )
            self.post_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim), nn.SiLU(),
                nn.Dropout(net_dropout), nn.Linear(hidden_dim, token_dim),
            )
        else:
            raise ValueError("projection_style must be paper or legacy")

    def forward(self, x, score_mask=None, score_bias=None):
        batch_size, seq_len, _ = x.shape

        norm_x = self.pre_norm(x)
        projected = self.pre_proj(norm_x)
        if isinstance(self.pre_proj, nn.Linear):
            projected = F.silu(projected)
        u, q, k, v = projected.chunk(4, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        if score_bias is not None:
            logits = logits + score_bias
        if score_mask is not None:
            logits = logits.masked_fill(~score_mask.unsqueeze(1), float("-inf"))

        attention = torch.softmax(logits, dim=-1)
        pooled = torch.matmul(attention, v)
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, seq_len, self.token_dim)

        gate = u if self.projection_style == "paper" else F.silu(u)
        return self.post_proj(self.attn_norm(pooled) * gate)


# Legacy name: the earlier template configs (``UltraHSTU_*``) instantiate the
# same aligned model.
UltraHSTU = HSTU
HSTUPaper = HSTU
