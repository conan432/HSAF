# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


from torch import nn
from .dot_product_attention import (
    ScaledDotProductAttention,
    padding_mask_to_sdpa,
    validate_head_dim,
)


class MultiHeadTargetAttention(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True,
                 attention_activation_type="SoftMax"):
        super(MultiHeadTargetAttention, self).__init__()
        if isinstance(attention_dim, (list, tuple)):
            attention_dim = attention_dim[0] if len(attention_dim) > 0 else input_dim
        if not use_qkvo:
            attention_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = validate_head_dim(
            attention_dim, num_heads, "MultiHeadTargetAttention"
        )
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
            # Fuse the K/V projections: both consume the same history tensor,
            # so one GEMM halves projection kernel launches.
            self.W_kv = nn.Linear(input_dim, 2 * attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(
            dropout_rate=dropout_rate,
            attention_activation_type=attention_activation_type,
        )

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key, value = self.W_kv(history_sequence).chunk(2, dim=-1)
        else:
            query, key, value = target_item, history_sequence, history_sequence

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        mask, has_history = padding_mask_to_sdpa(
            mask, batch_size=batch_size, length=key.size(-2)
        )
        if mask is not None:
            mask = mask.expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        if has_history is not None:
            output = output * has_history.to(output.dtype)
        return output
