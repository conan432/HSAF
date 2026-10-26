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

import math

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers.blocks import MLP_Block


class GroupedChunkTokenizer(nn.Module):
    """Chunk tokenizer that keeps **semantic group** boundaries intact.

    The flat feature vector of these models is a concatenation of groups with
    different meaning, e.g. ``[non-sequence features | sequence summary]``.
    ``ChunkTokenizer`` splits that vector into ``num_tokens`` equal chunks, so a
    chunk can straddle two groups: with Taobao's 432-dim input and 26 chunks
    (chunk_dim = 17) the sequence summary (flat ``[320, 432)``) ends up spread
    over chunks 18-25 and the first of them also carries target features.
    A token produced that way cannot be attributed to a group, which makes any
    "inject/read this information only in the sequence tokens" design
    ill-defined.

    This tokenizer instead

    1. allocates the **config-fixed** total token budget to the groups
       proportionally to their embedding width (largest-remainder rounding,
       every group keeps at least one token), and
    2. chunks *inside* each group, padding per group when a group's width is not
       divisible by its token count.

    Attributes
    ----------
    group_dims : list[int]
        Width of each group in the flat input, in the order they are
        concatenated.
    group_num_tokens : list[int]
        Tokens allocated to each group (sums to ``num_tokens``).
    token_slices : list[tuple[int, int]]
        ``[start, stop)`` of the token rows that belong to each group, so the
        model can address "the sequence tokens" / "the non-sequence tokens"
        exactly.
    """

    def __init__(self, group_dims, token_dim, num_tokens, activation="SiLU",
                 layer_norm=True):
        super(GroupedChunkTokenizer, self).__init__()
        group_dims = [int(d) for d in group_dims]
        if not group_dims or any(d <= 0 for d in group_dims):
            raise ValueError(f"group_dims must be positive, got {group_dims}")
        if num_tokens < len(group_dims):
            raise ValueError(
                f"num_tokens={num_tokens} is smaller than the number of groups "
                f"({len(group_dims)}); every group needs at least one token.")

        self.input_dim = int(sum(group_dims))
        self.num_tokens = int(num_tokens)
        self.group_dims = group_dims
        self.group_num_tokens = self._allocate(group_dims, self.num_tokens)
        self.group_chunk_dims = [
            int(math.ceil(dim / n)) for dim, n in zip(group_dims, self.group_num_tokens)
        ]
        self.group_pad_dims = [
            chunk * n - dim
            for chunk, n, dim in zip(self.group_chunk_dims,
                                     self.group_num_tokens, group_dims)
        ]
        # token row ranges per group, in group order
        slices, start = [], 0
        for n in self.group_num_tokens:
            slices.append((start, start + n))
            start += n
        self.token_slices = slices
        # flat feature ranges per group, in group order
        feat_slices, start = [], 0
        for dim in group_dims:
            feat_slices.append((start, start + dim))
            start += dim
        self.feature_slices = feat_slices

        self.token_mlps = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(input_dim=chunk_dim,
                          hidden_units=[token_dim],
                          hidden_activations=activation,
                          layer_norm=layer_norm)
                for _ in range(n)
            ])
            for chunk_dim, n in zip(self.group_chunk_dims, self.group_num_tokens)
        ])

    @staticmethod
    def _allocate(group_dims, num_tokens):
        """Largest-remainder allocation, proportional to the group widths."""
        total = float(sum(group_dims))
        exact = [dim / total * num_tokens for dim in group_dims]
        alloc = [max(1, int(math.floor(value))) for value in exact]
        # hand out / take back the remainder, always keeping >= 1 per group
        while sum(alloc) < num_tokens:
            order = sorted(range(len(group_dims)),
                           key=lambda i: -(exact[i] - math.floor(exact[i])))
            for index in order:
                if sum(alloc) >= num_tokens:
                    break
                alloc[index] += 1
        while sum(alloc) > num_tokens:
            order = sorted(range(len(group_dims)),
                           key=lambda i: (alloc[i] - exact[i]), reverse=True)
            for index in order:
                if sum(alloc) <= num_tokens:
                    break
                if alloc[index] > 1:
                    alloc[index] -= 1
        return alloc

    def forward(self, feature_embeddings):
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
        if feature_embeddings.size(-1) != self.input_dim:
            raise ValueError(
                "GroupedChunkTokenizer expects input_dim={}, got {}.".format(
                    self.input_dim, feature_embeddings.size(-1)))
        tokens = []
        for group, (feat_lo, feat_hi) in enumerate(self.feature_slices):
            group_input = feature_embeddings[:, feat_lo:feat_hi]
            pad_dim = self.group_pad_dims[group]
            if pad_dim > 0:
                group_input = F.pad(group_input, (0, pad_dim))
            chunks = group_input.view(-1, self.group_num_tokens[group],
                                      self.group_chunk_dims[group])
            tokens.extend([
                mlp(chunks[:, i, :])
                for i, mlp in enumerate(self.token_mlps[group])
            ])
        return torch.stack(tokens, dim=1)
