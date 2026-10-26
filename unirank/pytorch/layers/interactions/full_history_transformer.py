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
"""Full-history Transformer encoder shared by RankMixer/HeMix variants."""

import torch
from torch import nn

from unirank.pytorch.layers.attentions import ScaledDotProductAttention


class FullHistoryTransformerEncoder(nn.Module):
    """Bidirectional self-attention over the complete behavior history.

    OneTrans and HyFormer compare RankMixer with a Full Transformer sequence
    module.  This module is the shared implementation of that controlled
    sequence encoder, used by ``RankMixer_fullseq`` and ``HeMix_fullseq``.
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        expansion_factor=2,
        net_dropout=0.0,
    ):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.attn_norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.qkv_proj = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate=net_dropout)
        hidden_dim = input_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

    def forward(self, sequence_tokens, mask=None):
        if mask is not None:
            sequence_tokens = sequence_tokens * mask.unsqueeze(-1).to(
                sequence_tokens.dtype
            )
        residual = sequence_tokens
        normed = self.attn_norm(sequence_tokens)
        batch_size, length, dim = normed.shape
        attention_mask = None
        has_history = None
        if mask is not None:
            attention_mask = mask.bool().clone()
            has_history = attention_mask.sum(dim=-1, keepdim=True).gt(0)
            if length > 0 and has_history.logical_not().any():
                # Projections have no bias, so this exposed padding key/value
                # remains zero and avoids an all-masked SDPA row.
                attention_mask[has_history.logical_not().squeeze(1), 0] = True
            attention_mask = attention_mask.view(batch_size, 1, 1, length)

        def split_heads(x):
            return x.view(
                batch_size, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = self.qkv_proj(normed).chunk(3, dim=-1)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        attn_out, _ = self.dot_attention(
            q,
            k,
            v,
            scale=self.head_dim ** 0.5,
            mask=attention_mask,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, length, dim
        )
        attn_out = self.out_proj(attn_out)
        if has_history is not None:
            attn_out = attn_out * has_history.unsqueeze(1).to(attn_out.dtype)
        sequence_tokens = residual + attn_out
        if mask is not None:
            sequence_tokens = sequence_tokens * mask.unsqueeze(-1).to(
                sequence_tokens.dtype
            )
        sequence_tokens = sequence_tokens + self.ffn(
            self.ffn_norm(sequence_tokens)
        )
        if mask is not None:
            sequence_tokens = sequence_tokens * mask.unsqueeze(-1).to(
                sequence_tokens.dtype
            )
        return sequence_tokens

