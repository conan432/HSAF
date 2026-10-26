# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2018. pengshuang@Github for ScaledDotProductAttention.
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
import torch.nn.functional as F
from torch import nn
from unirank.pytorch.torch_utils import get_activation


def safe_padding_mask(mask, batch_size=None, length=None):
    """Return a boolean padding mask with no all-masked rows.

    SDPA is undefined for rows whose keys are all masked.  The models in this
    repository keep padded sequence tokens zeroed, so exposing one padding
    position only for genuinely empty rows preserves the result while keeping
    the fused attention kernels finite.
    """
    if mask is None:
        return None
    safe = mask.bool().clone()
    if batch_size is not None and length is not None:
        if safe.shape[:2] != (batch_size, length):
            raise ValueError(
                "attention mask must have shape [{}, {}], got {}".format(
                    batch_size, length, tuple(safe.shape)
                )
            )
    if safe.size(1) > 0:
        empty_rows = safe.sum(dim=-1) == 0
        if empty_rows.any():
            safe[empty_rows, 0] = True
    return safe


def padding_mask_to_sdpa(mask, batch_size=None, length=None):
    """Convert a [B, L] padding mask to a broadcastable SDPA bool mask.

    Returns ``(sdpa_mask, has_history)`` where ``sdpa_mask`` has shape
    ``[B, 1, 1, L]`` and ``has_history`` has shape ``[B, 1]``.
    """
    safe = safe_padding_mask(mask, batch_size=batch_size, length=length)
    if safe is None:
        return None, None
    has_history = safe.any(dim=-1, keepdim=True)
    return safe.view(safe.size(0), 1, 1, safe.size(1)), has_history


def validate_head_dim(input_dim, num_heads, module_name="attention"):
    """Validate and return the per-head dimension."""
    input_dim = int(input_dim)
    num_heads = int(num_heads)
    if num_heads <= 0:
        raise ValueError(
            f"{module_name}: num_heads must be positive, got {num_heads}"
        )
    if input_dim % num_heads != 0:
        raise ValueError(
            f"{module_name}: input_dim={input_dim} must be divisible by "
            f"num_heads={num_heads}"
        )
    return input_dim // num_heads


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with configurable score activation."""

    SUPPORTED_ACTIVATIONS = {
        "softmax": "SoftMax",
        "none": "None",
        "relu": "ReLU",
        "softplus": "SoftPlus",
        "silu": "SiLU",
        "gelu": "GeLU",
        "sigmoid": "Sigmoid",
        "mish": "Mish",
    }

    def __init__(self, dropout_rate=0., attention_activation_type="SoftMax"):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        activation_key = (
            "none"
            if attention_activation_type is None
            else str(attention_activation_type).strip().lower()
        )
        if activation_key not in self.SUPPORTED_ACTIVATIONS:
            supported = ", ".join(self.SUPPORTED_ACTIVATIONS.values())
            raise ValueError(
                "Unsupported attention_activation_type={!r}. Expected one of: {}"
                .format(attention_activation_type, supported)
            )
        self.attention_activation_type = self.SUPPORTED_ACTIVATIONS[activation_key]
        self.attention_activation = (
            None
            if self.attention_activation_type == "SoftMax"
            else get_activation(self.attention_activation_type)
        )

    def forward(self, Q, K, V, scale=None, mask=None, is_causal=False):
        # Bool mask: False means masked. Float mask: additive logits bias.
        if self.attention_activation_type == "SoftMax":
            attn_mask = mask
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(Q.dtype)
            if attn_mask is not None and is_causal:
                # SDPA accepts either an explicit mask or is_causal; combine
                # them explicitly so callers can use both safely.
                query_length, key_length = Q.size(-2), K.size(-2)
                causal = torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=Q.device,
                ).tril()
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & causal
                else:
                    causal_bias = torch.zeros_like(attn_mask)
                    causal_bias = causal_bias.masked_fill(
                        ~causal, torch.finfo(attn_mask.dtype).min
                    )
                    attn_mask = attn_mask + causal_bias
                is_causal = False
            sdpa_scale = None if scale is None else 1.0 / scale
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
                scale=sdpa_scale,
            )
            return output, None

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        if scale is not None:
            attention_scores = attention_scores / scale
        valid_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                valid_mask = mask
            else:
                attention_scores = attention_scores + mask.to(
                    attention_scores.dtype
                )
        attention_weights = attention_scores
        if self.attention_activation is not None:
            attention_weights = self.attention_activation(attention_scores)

        if is_causal:
            query_length, key_length = Q.size(-2), K.size(-2)
            causal_mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=Q.device,
            ).tril()
            valid_mask = causal_mask if valid_mask is None else valid_mask & causal_mask
        if valid_mask is not None:
            attention_weights = attention_weights.masked_fill(~valid_mask, 0.0)
        attention_weights = F.dropout(
            attention_weights,
            p=self.dropout_rate,
            training=self.training,
        )
        output = torch.matmul(attention_weights, V)
        return output, None

