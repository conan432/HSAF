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

import torch
from torch import nn
from unirank.pytorch.layers.blocks import MLP_Block


class RandomSplitTokenizer(nn.Module):
    def __init__(self, input_dim, token_dim, num_tokens, num_fields,
                 activation="SiLU", layer_norm=True):
        super(RandomSplitTokenizer, self).__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if num_fields < num_tokens:
            raise ValueError(
                "num_fields ({}) must be greater than or equal to num_tokens ({}).".format(
                    num_fields, num_tokens
                )
            )
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.register_buffer("field_permutation", torch.randperm(num_fields))
        base_group_size, remainder = divmod(num_fields, num_tokens)
        self.group_sizes = tuple(
            base_group_size + (1 if index < remainder else 0)
            for index in range(num_tokens)
        )
        self.token_mlps = nn.ModuleList([
            MLP_Block(
                input_dim=group_size * input_dim,
                hidden_units=[token_dim],
                hidden_activations=activation,
                layer_norm=layer_norm,
            )
            for group_size in self.group_sizes
        ])

    def forward(self, feature_embeddings):
        if feature_embeddings.dim() != 3:
            raise ValueError(
                "RandomSplitTokenizer expects a 3D tensor, got shape {}.".format(
                    tuple(feature_embeddings.shape)
                )
            )
        if (feature_embeddings.size(1) != self.num_fields or
                feature_embeddings.size(2) != self.input_dim):
            raise ValueError(
                "RandomSplitTokenizer expects shape [B, {}, {}], got {}.".format(
                    self.num_fields, self.input_dim, tuple(feature_embeddings.shape)
                )
            )
        shuffled_embeddings = feature_embeddings.index_select(
            dim=1,
            index=self.field_permutation,
        )
        groups = torch.split(shuffled_embeddings, self.group_sizes, dim=1)
        return torch.stack([
            mlp(group.flatten(start_dim=1))
            for group, mlp in zip(groups, self.token_mlps)
        ], dim=1)
