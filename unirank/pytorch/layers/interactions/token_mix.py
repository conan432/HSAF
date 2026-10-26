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
import torch.nn.functional as F

class MultiHeadTokenMixing(nn.Module):
    def __init__(self, input_dim, num_token):
        super(MultiHeadTokenMixing, self).__init__()
        self.num_token = num_token # = num_heads
        self.input_dim = input_dim
        self.pad_dim = (-input_dim) % num_token
        self.padded_input_dim = input_dim + self.pad_dim
        self.head_dim = self.padded_input_dim // num_token
        valid_mask = torch.arange(self.padded_input_dim) < input_dim
        valid_mask = valid_mask.expand(num_token, -1)
        valid_mask = valid_mask.reshape(
            num_token, num_token, self.head_dim
        ).transpose(0, 1).reshape(-1)
        self.register_buffer(
            "valid_index",
            valid_mask.nonzero(as_tuple=False).flatten(),
            persistent=False,
        )

    def forward(self, x):  # x: [B, T, D]
        if self.pad_dim == 0:
            # Fast path: token mixing is a pure head/token transpose when the
            # hidden dimension divides evenly by the token count. This avoids
            # tensor_split + index_select overhead.
            B, T, D = x.shape
            return (
                x.view(B, T, self.num_token, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(B, T, D)
            )
        if self.pad_dim:
            x = F.pad(x, (0, self.pad_dim))
        heads = torch.tensor_split(x, self.num_token, dim=-1)
        mixed = torch.stack(heads, dim=1).flatten(start_dim=2)
        mixed = mixed.flatten(start_dim=1).index_select(1, self.valid_index)
        return mixed.reshape(x.size(0), self.num_token, self.input_dim)
