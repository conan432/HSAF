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
"""Shared target/context/user grouped tokenizer."""

from typing import Dict, List, Sequence

import torch
from torch import nn

from .auto_split_tokenizer import AutoSplitTokenizer


CONTEXT_FIELD_CANDIDATES = (
    "day_of_week",
    "is_weekend",
    "hour",
    "tab",
    "pid",
    "session_id",
)


def split_non_sequence_fields(feature_map, embedding_dim=16) -> Dict[str, List[str]]:
    """Split feature_map fields into target/context/user groups."""
    target, context, user = [], [], []
    for feat, spec in feature_map.features.items():
        if feat in feature_map.labels or spec.get("type") == "meta":
            continue
        source = spec.get("source")
        if source == "item":
            target.append(feat)
        elif source == "action":
            # Action belongs to the behavior sequence, not the non-sequence
            # target/context/user tokenizer.
            continue
        elif feat in CONTEXT_FIELD_CANDIDATES:
            context.append(feat)
        else:
            user.append(feat)
    return {"target": target, "context": context, "user": user}


def group_dims(feature_map, groups, embedding_dim=16) -> List[int]:
    dims = []
    for name in ("target", "context", "user"):
        dims.append(sum(
            feature_map.features[feat].get("embedding_dim", embedding_dim)
            for feat in groups[name]
        ))
    return dims


def group_flat_slices(feature_map, groups, embedding_dim=16) -> Dict[str, List]:
    """Return flat-vector slices for each group.

    Target slices index the flattened item/action vector; context/user slices
    index the flattened non-item vector.
    """
    slices = {"target": [], "context": [], "user": []}
    item_pos = 0
    non_item_pos = 0
    for feat, spec in feature_map.features.items():
        if feat in feature_map.labels or spec.get("type") == "meta":
            continue
        dim = spec.get("embedding_dim", embedding_dim)
        source = spec.get("source")
        if source in ("item", "action"):
            if source == "item":
                slices["target"].append((item_pos, item_pos + dim))
            item_pos += dim
            continue
        if feat in CONTEXT_FIELD_CANDIDATES:
            slices["context"].append((non_item_pos, non_item_pos + dim))
        else:
            slices["user"].append((non_item_pos, non_item_pos + dim))
        non_item_pos += dim
    return slices


def gather_flat(tensor: torch.Tensor, slices: Sequence):
    """Gather and concatenate slices from a flat feature tensor."""
    parts = [tensor[:, start:end] for start, end in slices]
    if not parts:
        return tensor.new_zeros(tensor.size(0), 0)
    return torch.cat(parts, dim=-1)


def resolve_group_token_counts(
    dims: Sequence[int],
    requested: Sequence[int],
) -> List[int]:
    """Zero empty groups and redistribute their token budget.

    This keeps the total NS token count stable across datasets that lack a
    particular semantic group (for example QK has no context fields).
    """
    if len(dims) != len(requested):
        raise ValueError("dims and requested must have the same length")
    counts = [
        int(n) if int(dim) > 0 else 0
        for dim, n in zip(dims, requested)
    ]
    missing = int(sum(requested)) - int(sum(counts))
    if missing > 0:
        donor_order = [2, 0, 1]  # prefer user, then target, then context
        for idx in donor_order:
            if dims[idx] > 0:
                counts[idx] += missing
                break
    return counts


class GroupTokenizer(nn.Module):
    """Tokenize target/context/user groups with independent token counts.

    Each group uses its own ``AutoSplitTokenizer``; the outputs are
    concatenated in target -> context -> user order.  Groups with zero
    embedding dimension or zero token count are skipped.
    """

    def __init__(
        self,
        group_dims: Sequence[int],
        group_num_tokens: Sequence[int],
        token_dim: int,
        activation: str = "SiLU",
        layer_norm: bool = True,
    ):
        super().__init__()
        if len(group_dims) != len(group_num_tokens):
            raise ValueError("group_dims and group_num_tokens must align")
        self.group_dims = [int(d) for d in group_dims]
        self.group_num_tokens = [int(n) for n in group_num_tokens]
        self.tokenizers = nn.ModuleList()
        for dim, num_tokens in zip(self.group_dims, self.group_num_tokens):
            if dim == 0 or num_tokens == 0:
                self.tokenizers.append(nn.Identity())
            else:
                self.tokenizers.append(
                    AutoSplitTokenizer(
                        input_dim=dim,
                        token_dim=token_dim,
                        num_tokens=num_tokens,
                        activation=activation,
                        layer_norm=layer_norm,
                    )
                )

    @property
    def num_tokens(self):
        return int(sum(self.group_num_tokens))

    def forward(self, group_embeddings: Sequence[torch.Tensor]):
        if len(group_embeddings) != len(self.group_dims):
            raise ValueError("group_embeddings must align with group_dims")
        outputs = []
        for emb, dim, num_tokens, tokenizer in zip(
            group_embeddings,
            self.group_dims,
            self.group_num_tokens,
            self.tokenizers,
        ):
            if dim == 0 or num_tokens == 0:
                continue
            if emb.size(-1) != dim:
                raise ValueError(
                    f"group embedding dim mismatch: expected {dim}, "
                    f"got {emb.size(-1)}"
                )
            outputs.append(tokenizer(emb))
        if not outputs:
            raise ValueError("GroupTokenizer received no non-empty group")
        return torch.cat(outputs, dim=1)
