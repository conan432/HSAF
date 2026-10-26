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
"""History-state input ablation using the ordinary numeric non-sequence path.

This ablation asks whether the *unmodified* backbone can exploit history state
when the state is supplied in the same way as an ordinary numeric feature.
Each scalar is registered as a temporary numeric field and therefore goes
through the unmodified backbone's ordinary ``FeatureEmbedding`` path:

    scalar -> Linear(1, embedding_dim, bias=False)

The numeric-field embeddings are concatenated with the original user-context
and target embeddings before the original tokenizer.  The tokenizer therefore
sees one longer flattened non-sequence input, while its token count,
architecture, loss, read-out, and towers remain unchanged.

This module deliberately performs **no bucketing**.  There is no one-hot,
histogram, quantile/bin edge, bucket embedding, extra sequence token, gate,
calibration head, or attention bias.  Bucketing belongs exclusively to the
proposed HSAF method, not to this feature-engineering baseline.

The continuous scalars are:

* global state: effective history length, recency, and time span;
* local state : mean request-to-event gap, its standard deviation, and the mean
  normalized event position, where the most recent event has position 0.

All time quantities are converted to seconds.  Empty histories produce an
all-zero state vector, and padded events never contribute to any statistic.
"""

import torch

from .EST import EST
from .RankMixer import RankMixer
from .EviTime_gatecalib_use_time import _read_evitime_inputs, _resolve_time_unit


class StateFeatureMixin:
    """Encode request/event history-state summaries as numeric non-seq fields."""

    _GLOBAL_SCALARS = ("length", "recency", "span")
    _LOCAL_SCALARS = ("gap_mean", "gap_std", "pos_mean")
    _ALL_SCALARS = _GLOBAL_SCALARS + _LOCAL_SCALARS

    def _init_state_features(self, feature_map, embedding_dim,
                             state_global=True, state_local=True,
                             time_unit=None):
        self.state_use_global = bool(state_global)
        self.state_use_local = bool(state_local)
        if not (self.state_use_global or self.state_use_local):
            raise ValueError("at least one of state_global/state_local must be on")

        self.state_global_scalars = (
            self._GLOBAL_SCALARS if self.state_use_global else ()
        )
        self.state_local_scalars = (
            self._LOCAL_SCALARS if self.state_use_local else ()
        )
        self.state_scalar_names = (
            self.state_global_scalars + self.state_local_scalars
        )
        self.state_num_scalars = len(self.state_scalar_names)
        self.state_feature_dim = self.state_num_scalars * int(embedding_dim)
        self.feature_map = feature_map
        self.state_time_unit = _resolve_time_unit(self, time_unit)

        # Register synthetic numeric fields before the backbone constructor so
        # that its normal FeatureEmbedding owns their encoders and the original
        # tokenizer input width is computed without rebuilding either module.
        existing = set(feature_map.features)
        self.state_feature_names = tuple(
            f"__history_state_{name}" for name in self.state_scalar_names
        )
        collisions = existing.intersection(self.state_feature_names)
        if collisions:
            raise ValueError(
                f"history-state feature names collide with dataset fields: {collisions}"
            )
        for name in self.state_feature_names:
            feature_map.features[name] = {
                "active": True,
                "type": "numeric",
                "source": "user",
                "embedding_dim": int(embedding_dim),
            }

    def _time_scale(self):
        """Convert the configured timestamp unit to seconds."""
        if self.state_time_unit in {"ms", "millisecond", "milliseconds"}:
            return 1.0 / 1000.0
        if self.state_time_unit in {"us", "microsecond", "microseconds"}:
            return 1.0 / 1_000_000.0
        return 1.0

    def _all_state_values(self, mask, timestamp_seq, target_timestamp):
        """Return ``[B, 6]`` continuous values in seconds, without bucketing."""
        if mask.ndim != 2:
            raise ValueError(f"mask must be [B, T], got {tuple(mask.shape)}")
        batch_size, _ = mask.shape
        device = mask.device
        valid = mask.to(device=device, dtype=torch.bool)
        valid_float = valid.to(torch.float32)

        length = valid_float.sum(dim=1)
        recency = torch.zeros(batch_size, dtype=torch.float32, device=device)
        span = torch.zeros_like(recency)
        gap_mean = torch.zeros_like(recency)
        gap_std = torch.zeros_like(recency)

        if timestamp_seq is not None and timestamp_seq.numel() > 0:
            stamps = timestamp_seq.to(device=device, dtype=torch.float64)
            if stamps.ndim != 2 or stamps.shape != mask.shape:
                raise ValueError(
                    "timestamp_seq must match mask shape [B, T], got "
                    f"{tuple(stamps.shape)} vs {tuple(mask.shape)}"
                )

            has_history = valid.any(dim=1)
            latest = torch.where(
                valid, stamps, torch.full_like(stamps, float("-inf"))
            ).max(dim=1).values
            earliest = torch.where(
                valid, stamps, torch.full_like(stamps, float("inf"))
            ).min(dim=1).values
            latest = torch.where(has_history, latest, torch.zeros_like(latest))
            earliest = torch.where(has_history, earliest, torch.zeros_like(earliest))

            if target_timestamp is not None:
                request_time = target_timestamp.to(
                    device=device, dtype=torch.float64
                ).reshape(-1)
                if request_time.numel() == 1 and batch_size != 1:
                    request_time = request_time.expand(batch_size)
                if request_time.numel() != batch_size:
                    raise ValueError(
                        "target_timestamp must contain one value per request; "
                        f"got {request_time.numel()} for batch {batch_size}"
                    )
                request_time = torch.where(request_time > 0, request_time, latest)
            else:
                request_time = latest

            scale = self._time_scale()
            recency = (request_time - latest).clamp_min(0.0) * scale
            span = (latest - earliest).clamp_min(0.0) * scale
            recency = torch.where(has_history, recency, torch.zeros_like(recency))
            span = torch.where(has_history, span, torch.zeros_like(span))

            request_gap = (
                request_time.unsqueeze(1) - stamps
            ).clamp_min(0.0) * scale
            request_gap = torch.where(
                valid, request_gap, torch.zeros_like(request_gap)
            )
            gap_mean64 = request_gap.sum(dim=1) / length.clamp_min(1.0).to(torch.float64)
            centred = torch.where(
                valid,
                request_gap - gap_mean64.unsqueeze(1),
                torch.zeros_like(request_gap),
            )
            gap_std64 = torch.sqrt(
                centred.square().sum(dim=1)
                / length.clamp_min(1.0).to(torch.float64)
            )
            gap_mean = torch.where(
                has_history, gap_mean64.to(torch.float32), gap_mean
            )
            gap_std = torch.where(
                has_history, gap_std64.to(torch.float32), gap_std
            )

        # Rank events from most recent (0) to oldest (1).  This is a continuous
        # summary, not a discretised position bin.
        reversed_valid = torch.flip(valid.long(), dims=[1])
        rank_from_recent = torch.flip(
            torch.cumsum(reversed_valid, dim=1) - 1, dims=[1]
        ).to(torch.float32)
        position = rank_from_recent / (length - 1.0).clamp_min(1.0).unsqueeze(1)
        position = torch.where(valid, position, torch.zeros_like(position))
        pos_mean = (position * valid_float).sum(dim=1) / length.clamp_min(1.0)

        return torch.stack([
            length,
            recency.to(torch.float32),
            span,
            gap_mean,
            gap_std,
            pos_mean,
        ], dim=-1).to(torch.float32)

    def _state_values(self, mask, timestamp_seq, target_timestamp):
        """Select the enabled global/local scalars in a fixed order."""
        all_values = self._all_state_values(mask, timestamp_seq, target_timestamp)
        if self.state_scalar_names == self._ALL_SCALARS:
            return all_values
        indices = [self._ALL_SCALARS.index(name) for name in self.state_scalar_names]
        return all_values.index_select(
            1, torch.tensor(indices, device=all_values.device, dtype=torch.long)
        )

    def _encode_state_values(self, values):
        """Encode with the numeric layers actually owned by FeatureEmbedding."""
        if values.size(1) != len(self.state_feature_names):
            raise RuntimeError(
                f"expected {len(self.state_feature_names)} state values, "
                f"got {values.size(1)}"
            )
        layers = self.embedding_layer.embedding_layer.embedding_layers
        return torch.cat([
            layers[name](values[:, i:i + 1])
            for i, name in enumerate(self.state_feature_names)
        ], dim=-1)

    def _state_feature_embeddings(self, mask, timestamp_seq, target_timestamp):
        """Return ``[B, num_state_scalars * embedding_dim]``."""
        values = self._state_values(mask, timestamp_seq, target_timestamp)
        return self._encode_state_values(values)

    def _augment_batch_with_state(self, batch_dict, mask, timestamp_seq,
                                  target_timestamp):
        """Add state values as ordinary numeric fields to the batch."""
        values = self._state_values(mask, timestamp_seq, target_timestamp)
        augmented = dict(batch_dict)
        for i, name in enumerate(self.state_feature_names):
            augmented[name] = values[:, i]
        return augmented


class RankMixer_StateFeature(StateFeatureMixin, RankMixer):
    """Decoupled backbone plus history state on the numeric non-seq path."""

    def __init__(self, feature_map, model_id="RankMixer_StateFeature",
                 num_heads=2, state_global=True, state_local=True,
                 time_unit=None, **kwargs):
        self._init_state_features(
            feature_map=feature_map,
            embedding_dim=kwargs.get("embedding_dim", 10),
            state_global=state_global,
            state_local=state_local,
            time_unit=time_unit,
        )
        super().__init__(feature_map, model_id=model_id, **kwargs)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_evitime_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]
        sequence_tokens = self.item_token_proj(sequence_emb)

        batch_dict = self._augment_batch_with_state(
            batch_dict, mask, timestamp_seq, target_timestamp
        )
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)

        unified_tokens = self.unified_tokenizer_layer(
            feature_embeddings, sequence_tokens, mask
        )
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers, unified_tokens
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        for i, label in enumerate(self.feature_map.labels):
            return_dict[f"{label}_pred"] = y_pred[i]
        return return_dict


class EST_StateFeature(StateFeatureMixin, EST):
    """Joint backbone plus history state on the numeric non-seq path."""

    def __init__(self, feature_map, model_id="EST_StateFeature",
                 num_heads=2, state_global=True, state_local=True,
                 time_unit=None, **kwargs):
        self._init_state_features(
            feature_map=feature_map,
            embedding_dim=kwargs.get("embedding_dim", 10),
            state_global=state_global,
            state_local=state_local,
            time_unit=time_unit,
        )
        super().__init__(feature_map, model_id=model_id, **kwargs)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs[:4])
        timestamp_seq, target_timestamp = _read_evitime_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :].view(batch_size, -1, self.embedding_dim)
        sequence_emb = item_seq_emb[:, 0:-1, :]
        s_tokens = self.item_token_proj(sequence_emb)

        batch_dict = self._augment_batch_with_state(
            batch_dict, mask, timestamp_seq, target_timestamp
        )
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([
            user_context_emb,
            target_emb.reshape(batch_size, -1),
        ], dim=-1)
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)

        s_tokens, unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            unified_tokens,
            mask,
        )

        ns_repr = (unified_tokens.mean(dim=1) if self.readout == "mean"
                   else unified_tokens.flatten(start_dim=1))
        bottom_output = torch.cat([
            ns_repr,
            self.masked_avg_pooling(s_tokens, mask),
        ], dim=-1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        for i, label in enumerate(self.feature_map.labels):
            return_dict[f"{label}_pred"] = y_pred[i]
        return return_dict


__all__ = ["RankMixer_StateFeature", "EST_StateFeature"]
