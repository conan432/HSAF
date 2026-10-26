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


class FieldWiseTokenizer(nn.Module):
    def __init__(self, input_dim, token_dim, num_tokens, activation="SiLU",
                 layer_norm=True):
        super(FieldWiseTokenizer, self).__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.token_mlps = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=[token_dim],
                hidden_activations=activation,
                layer_norm=layer_norm,
            )
            for _ in range(num_tokens)
        ])

    def forward(self, token_inputs):
        if token_inputs.dim() != 3:
            raise ValueError(
                "FieldWiseTokenizer expects a 3D tensor, got shape {}.".format(
                    tuple(token_inputs.shape)
                )
            )
        if token_inputs.size(1) != self.num_tokens or token_inputs.size(2) != self.input_dim:
            raise ValueError(
                "FieldWiseTokenizer expects shape [B, {}, {}], got {}.".format(
                    self.num_tokens, self.input_dim, tuple(token_inputs.shape)
                )
            )
        return torch.stack([
            mlp(token_inputs[:, i, :])
            for i, mlp in enumerate(self.token_mlps)
        ], dim=1)
