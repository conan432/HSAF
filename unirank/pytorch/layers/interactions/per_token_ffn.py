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
from unirank.pytorch.torch_utils import get_activation

class PerTokenFeedForward(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0,
                 hidden_activation="GELU"):
        super(PerTokenFeedForward, self).__init__()
        self.input_dim = input_dim
        self.num_token = num_token
        self.hidden_dim = expand * input_dim
        self.W1 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.input_dim))
        self.act = get_activation(hidden_activation)
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        h = torch.einsum("btd,tdk->btk", x, self.W1) + self.b1
        h = self.act(h)
        h = self.dropout(h)
        y = torch.einsum("btk,tkd->btd", h, self.W2) + self.b2
        return y

class PerTokenSwiGLU(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0):
        super(PerTokenSwiGLU, self).__init__()
        self.input_dim = input_dim
        self.num_token = num_token
        self.hidden_dim = expand * input_dim
        self.W1 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.hidden_dim))
        self.W3 = nn.Parameter(torch.empty(self.num_token, self.hidden_dim, self.input_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        h1 = torch.einsum("btd,tdk->btk", x, self.W1) + self.b1
        h1 = self.act(h1)
        h1 = self.dropout(h1)
        h2 = torch.einsum("btd,tdk->btk", x, self.W2) + self.b2
        y = torch.einsum("btk,tkd->btd", h1 * h2, self.W3)
        return y

class SwiGLU(nn.Module):
    def __init__(self, input_dim, expand=4, net_dropout=0.0):
        super(SwiGLU, self).__init__()
        hidden_dim = int(input_dim * expand)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

    def forward(self, x): # x: [B, T, D]
        y = self.act(self.W1(x)) * self.W2(x)
        y = self.dropout(y)
        y = self.W3(y)
        return y

class SparseMoELayer(nn.Module):
    """ReLU-routed per-token MoE used by RankMixer.

    ``routing_mode="relu"`` follows Eq. (10) and returns the ReLU-gate L1
    regularizer in Eq. (11).  ``routing_mode="dtsi"`` additionally keeps a
    dense training router and a ReLU inference router as described by
    DTSI-MoE.  The current PyTorch implementation materializes all expert
    outputs; it preserves the routing semantics but does not provide the
    paper's kernel-level sparse-inference throughput gain.
    """
    def __init__(self,
                 input_dim,
                 num_token,
                 expand=2,
                 num_experts=4,
                 net_dropout=0.0,
                 routing_mode="dtsi"):
        super(SparseMoELayer, self).__init__()
        self.input_dim = input_dim          # D
        self.num_token = num_token        # T
        self.hidden_dim = expand * input_dim  # kD
        self.num_experts = num_experts      # E
        self.routing_mode = str(routing_mode).strip().lower()
        if self.routing_mode not in {"relu", "dtsi"}:
            raise ValueError(
                "SparseMoELayer routing_mode must be 'relu' or 'dtsi', "
                f"got {routing_mode!r}"
            )

        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

        self.W1 = nn.Parameter(torch.empty(self.num_token, self.num_experts, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_token, self.num_experts, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_token, self.num_experts, self.hidden_dim, self.input_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_token, self.num_experts, self.input_dim))

        self.hidden_act = nn.GELU()
        self.gate_act = nn.ReLU()
        self.train_router = None
        if self.routing_mode == "dtsi":
            # Dense softmax routing gives every expert a training gradient.
            self.train_router = nn.Parameter(
                torch.empty(self.num_token, self.num_experts, self.input_dim)
            )

        # Always kept separate from the dense router so eval can switch to the
        # L1-regularized ReLU router without changing training parameters.
        self.router = nn.Parameter(
            torch.empty(self.num_token, self.num_experts, self.input_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.xavier_uniform_(self.router)
        if self.train_router is not None:
            nn.init.xavier_uniform_(self.train_router)

    def _ffn_all_experts(self, x: torch.Tensor):
        """
        x: [B, T, D] -> expert_out: [B, T, E, D]
        """
        h = torch.einsum("btd,tedk->btek", x, self.W1) + self.b1  # [B, T, E, kD]
        h = self.hidden_act(h)
        h = self.dropout(h)
        expert_out = torch.einsum("btek,tekd->bted", h, self.W2) + self.b2  # [B, T, E, D]
        return expert_out

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        if T != self.num_token or D != self.input_dim:
            raise ValueError(
                "SparseMoELayer input shape mismatch: "
                f"expected [{B}, {self.num_token}, {self.input_dim}], got {list(x.shape)}"
            )

        expert_out = self._ffn_all_experts(x)                         # [B,T,E,D]
        infer_gates = self.gate_act(
            torch.einsum("btd,ted->bte", x, self.router)
        )  # [B,T,E]
        # Eq. (11) is defined per sample; average over the mini-batch so the
        # coefficient does not implicitly scale with batch_size.
        l1_regularizer = infer_gates.sum(dim=(1, 2)).mean()

        if self.training and self.train_router is not None:
            gates = torch.softmax(
                torch.einsum("btd,ted->bte", x, self.train_router), dim=-1
            )
        else:
            gates = infer_gates
        out = torch.sum(gates.unsqueeze(-1) * expert_out, dim=2)  # [B,T,D]
        return out, l1_regularizer
