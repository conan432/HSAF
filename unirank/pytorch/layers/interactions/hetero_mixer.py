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
"""Shared building blocks for the mixer family (RankMixer / UniMixer / HeMix).

The mixer baselines treat the behaviour sequence as an input feature domain and
do not emphasise sequence modelling, so the whole family is aligned on HeMix's
history reading recipe:

* non-sequence features (user context + target) are split into ``N`` tokens by
  ``AutoSplitTokenizer`` -- these are the *dynamic* queries;
* ``N`` learnable *fixed* (CLS) queries are appended;
* every dynamic/fixed query reads the history through ``MixedHeteroAttention``
  (token-specific ``W_q``, shared ``K``/``V``), producing ``2N`` history tokens;
* the resulting ``3N`` tokens (``2N`` history + ``N`` non-sequence) are then
  mixed layer by layer.

HeMix keeps its long/short-term split by reading two windows (the most recent
``real_seq_len`` items and the rest) with two disjoint group of queries; the
other two mixers read a single window covering the whole history.
"""

import torch
from torch import nn

from ..attentions import ScaledDotProductAttention
from ..tokenization import AutoSplitTokenizer
from .per_token_ffn import PerTokenSwiGLU


class MixedHeteroAttention(nn.Module):
    """History reader: token-specific queries attend over a shared sequence.

    ``ns_tokens`` holds one query row per reading query (dynamic feature tokens
    first, then the learnable fixed queries); ``s_tokens`` is the behaviour
    sequence that is read.  ``W_q`` is token-specific while ``K``/``V`` are
    shared, which is the "per-token W_q" convention the mixer family is aligned
    on.
    """

    def __init__(self, input_dim, num_heads, num_ns_token):
        super(MixedHeteroAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_s.weight)
        nn.init.xavier_uniform_(self.W_v_s.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, s_tokens, ns_tokens):
        B, Ls, D = s_tokens.shape
        _, Lns, _ = ns_tokens.shape
        if Ls == 0:
            return ns_tokens

        q_weight = self.W_q_ns[:Lns]
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, q_weight)
        k_s = self.W_k_s(s_tokens)
        v_s = self.W_v_s(s_tokens)
        q = q_ns.view(B, Lns, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_s.view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
        output, _ = self.dot_attention(q, k, v, scale=self.head_dim ** 0.5)
        output = output.transpose(1, 2).contiguous().view(B, Lns, D)
        return self.W_o(output)


class HeteroQueryTokenizer(nn.Module):
    """HeMix-style history reader shared by the mixer family.

    Args:
        input_dim: width of the flattened ``[user context || target]`` vector.
        token_dim: hidden size of the interaction layers.
        num_ns_token: ``N``, number of non-sequence (dynamic) tokens.
        num_real_token: ``r``.  ``r > 0`` splits the history into a recent
            window (last ``real_seq_len`` items, read by ``r`` dynamic + ``r``
            fixed queries) and an earlier window (read by the remaining
            ``N - r`` dynamic + fixed queries) -- HeMix's long/short-term
            design.  ``r == 0`` reads a single window covering the whole
            history.
        num_heads: attention heads of the history reader.
        real_seq_len: length of the recent window when ``num_real_token > 0``.

    Returns ``[B, 3N, D]``: ``[2N history tokens || N non-sequence tokens]``.
    """

    def __init__(self,
                 input_dim,
                 token_dim,
                 num_ns_token,
                 num_real_token=0,
                 num_heads=2,
                 real_seq_len=20,
                 activation="SiLU",
                 layer_norm=True):
        super(HeteroQueryTokenizer, self).__init__()
        self.num_ns_token = int(num_ns_token)
        self.num_real_token = int(num_real_token)
        self.num_glocal_token = self.num_ns_token - self.num_real_token
        self.real_seq_len = int(real_seq_len)
        if self.num_ns_token <= 0:
            raise ValueError("num_ns_token must be positive.")
        if not 0 <= self.num_real_token <= self.num_ns_token:
            raise ValueError(
                "num_real_token must be within [0, num_ns_token], got {}.".format(
                    self.num_real_token
                )
            )

        self.token_mlps = AutoSplitTokenizer(
            input_dim=input_dim,
            token_dim=token_dim,
            num_tokens=self.num_ns_token,
            activation=activation,
            layer_norm=layer_norm,
        )

        if self.num_real_token > 0:
            self.cls_tokens_glocal = nn.Parameter(
                torch.empty(1, self.num_glocal_token, token_dim)
            )
            self.cls_tokens_real = nn.Parameter(
                torch.empty(1, self.num_real_token, token_dim)
            )
            self.mha_glocal = MixedHeteroAttention(
                input_dim=token_dim,
                num_heads=num_heads,
                num_ns_token=self.num_glocal_token * 2,
            )
            self.mha_real = MixedHeteroAttention(
                input_dim=token_dim,
                num_heads=num_heads,
                num_ns_token=self.num_real_token * 2,
            )
        else:
            self.cls_tokens = nn.Parameter(torch.empty(1, self.num_ns_token, token_dim))
            self.mha_full = MixedHeteroAttention(
                input_dim=token_dim,
                num_heads=num_heads,
                num_ns_token=self.num_ns_token * 2,
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_real_token > 0:
            nn.init.xavier_uniform_(self.cls_tokens_glocal)
            nn.init.xavier_uniform_(self.cls_tokens_real)
        else:
            nn.init.xavier_uniform_(self.cls_tokens)

    def forward(self, feature_embeddings, sequence_emb, mask=None):
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
        batch_size = feature_embeddings.size(0)
        ns_tokens = self.token_mlps(feature_embeddings)          # B x N x D

        if self.num_real_token > 0:
            real_emb = sequence_emb[:, -self.real_seq_len:, :]
            glocal_emb = sequence_emb[:, :-self.real_seq_len, :]
            if mask is not None:
                real_mask = mask[:, -self.real_seq_len:]
                glocal_mask = mask[:, :-self.real_seq_len]
                real_emb = real_emb * real_mask.unsqueeze(-1).float()
                glocal_emb = glocal_emb * glocal_mask.unsqueeze(-1).float()

            ns_glocal = ns_tokens[:, :self.num_glocal_token, :]
            ns_real = ns_tokens[:, self.num_glocal_token:, :]
            glocal_queries = torch.cat(
                [ns_glocal, self.cls_tokens_glocal.expand(batch_size, -1, -1)], dim=1
            )
            real_queries = torch.cat(
                [ns_real, self.cls_tokens_real.expand(batch_size, -1, -1)], dim=1
            )
            history_real = self.mha_real(s_tokens=real_emb, ns_tokens=real_queries)
            history_glocal = self.mha_glocal(s_tokens=glocal_emb, ns_tokens=glocal_queries)
            history_tokens = torch.cat([history_real, history_glocal], dim=1)
        else:
            if mask is not None:
                sequence_emb = sequence_emb * mask.unsqueeze(-1).float()
            queries = torch.cat(
                [ns_tokens, self.cls_tokens.expand(batch_size, -1, -1)], dim=1
            )
            history_tokens = self.mha_full(s_tokens=sequence_emb, ns_tokens=queries)

        return torch.cat([history_tokens, ns_tokens], dim=1)


class LowRankTokenMixing(nn.Module):
    """HeMix's heterogeneous mixing: head-wise low-rank token interaction."""

    def __init__(self, input_dim, num_token, num_heads=4, low_rank_dim=None):
        super(LowRankTokenMixing, self).__init__()
        assert input_dim % num_heads == 0, \
            "input_dim must be divisible by num_heads"

        self.num_token = num_token
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.mix_dim = num_token * self.head_dim
        if low_rank_dim is None:
            low_rank_dim = max(16, self.mix_dim // 8)
        self.low_rank_dim = low_rank_dim
        self.W_l = nn.Parameter(torch.empty(num_heads, self.mix_dim, low_rank_dim))
        self.b_l = nn.Parameter(torch.zeros(num_heads, low_rank_dim))
        self.W_r = nn.Parameter(torch.empty(num_heads, low_rank_dim, self.mix_dim))
        self.b_r = nn.Parameter(torch.zeros(num_heads, self.mix_dim))
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.zeros_(self.b_l)
        nn.init.zeros_(self.b_r)

    def forward(self, x, mix_gate=None, hist_slice=None):
        """x: [B, T, D] -> [B, T, D]

        ``mix_gate`` (M-A, "gate inside the token mixing"): a request-level
        scalar that scales the **history rows of the mixing input** before the
        low-rank mixing runs, i.e.

            x_in = [ g * x_hist , x_ns ]  ->  LowRankTokenMixing(x_in)

        so that the same ``g`` modulates every downstream flow of history
        (history -> NS *and* history -> history).  The residual outside the
        operator keeps the *ungated* representation, so the gate never writes
        into the residual stream; it only re-weights what the operator reads.
        ``hist_slice`` is the half-open row range of the history tokens.
        Defaults (``None``) reproduce the plain operator exactly.
        """
        B, T, D = x.shape
        assert T == self.num_token, \
            "token number mismatch: expected {}, got {}".format(self.num_token, T)
        if mix_gate is not None and hist_slice is not None:
            lo, hi = hist_slice
            if lo != hi:
                g = mix_gate
                while g.dim() < x.dim():
                    g = g.unsqueeze(0) if g.dim() == 0 else g.unsqueeze(-1)
                x = torch.cat([x[:, :lo, :], x[:, lo:hi, :] * g, x[:, hi:, :]],
                              dim=1)
        mixed = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mixed = mixed.reshape(B, self.num_heads, self.mix_dim)
        hidden = torch.einsum("bhf,hfr->bhr", mixed, self.W_l) + self.b_l.unsqueeze(0)
        hidden = self.act(hidden)
        out = torch.einsum("bhr,hrf->bhf", hidden, self.W_r) + self.b_r.unsqueeze(0)
        out = out.view(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return out.view(B, T, D)


class HeteroMixer(nn.Module):
    """HeMix block: ``[low-rank token mixing + AddNorm] -> [per-token FFN + AddNorm]``.

    The FFN operator is the per-token SwiGLU shared with the whole mixer family;
    the normalisation placement (pre-norm in both sub-layers) follows HeMix and
    RankMixer.
    """

    def __init__(self,
                 input_dim,
                 num_token,
                 num_layers,
                 expand=4,
                 num_mix_heads=4,
                 low_rank_dim=None,
                 net_dropout=0.0):
        super(HeteroMixer, self).__init__()
        self.num_layers = num_layers
        self.mixer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.pffn_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.mixer_layers = nn.ModuleList([
            LowRankTokenMixing(
                input_dim=input_dim,
                num_token=num_token,
                num_heads=num_mix_heads,
                low_rank_dim=low_rank_dim
            )
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenSwiGLU(
                input_dim=input_dim,
                num_token=num_token,
                expand=expand,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mix_gates=None, hist_slice=None):
        """``mix_gates``: optional per-layer gates (list/stack of ``[B,1,1]``).

        Each layer's gate scales the history rows of that layer's *mixing
        input* (see ``LowRankTokenMixing.forward``); ``None`` keeps the plain
        mixer behaviour, so base models are unaffected.
        """
        for i in range(self.num_layers):
            gate = None
            if mix_gates is not None:
                if torch.is_tensor(mix_gates):
                    # stacked per-layer gates: [L, B, 1, 1]; anything with fewer
                    # dims is a single gate shared by all layers
                    gate = mix_gates[i] if mix_gates.dim() >= 4 else mix_gates
                else:
                    gate = mix_gates[i]      # list / tuple of per-layer gates
            x = self.mixer_layers[i](self.mixer_norms[i](x), mix_gate=gate,
                                     hist_slice=hist_slice) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x
