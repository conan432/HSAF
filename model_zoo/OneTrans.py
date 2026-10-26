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
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import build_unified_tokenizer


class OneTrans(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="OneTrans",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 tokenizer_type="Auto",
                 attention_activation_type="SoftMax",
                 reduction_ratio=0.5,
                 readout="flatten",
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(OneTrans, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)

        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        # ``flatten`` (OneTrans' native readout) feeds every non-sequence token
        # into the tower; ``mean`` mean-pools them first.
        readout = str(readout).strip().lower()
        if readout not in ("flatten", "mean"):
            raise ValueError("readout must be 'flatten' or 'mean', got {!r}".format(readout))
        self.readout = readout

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.non_item_dim = 0
        self.num_item_fields = 0
        self.num_non_item_fields = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.num_item_fields += 1
            else:
                self.non_item_dim += emb_dim
                self.num_non_item_fields += 1

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Non-sequential feature tokenizer: generates num_ns_token NS tokens, because the target item is also regarded as an NS token, so self.item_info_dim is added
        tokenizer_input_dim = self.non_item_dim + self.item_info_dim
        self.num_tokenizer_fields = self.num_non_item_fields + self.num_item_fields
        (self.unified_tokenizer_layer,
         self.num_ns_token,
         self.tokenizer_uses_field_input) = build_unified_tokenizer(
            tokenizer_type=tokenizer_type,
            input_dim=tokenizer_input_dim,
            field_dim=embedding_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token,
            num_fields=self.num_tokenizer_fields,
        )
        self.tokenizer_type = str(tokenizer_type).strip().title()

        # Project sequence and target items to token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # OneTrans backbone
        self.unified_interaction_layers = OneTransBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=self.num_ns_token,
            dnn_activations=dnn_activations,
            expansion_factor=expansion_factor,
            attention_activation_type=attention_activation_type,
            reduction_ratio=reduction_ratio,
        )

        # Finally, only NS tokens are used for prediction.
        tower_input_dim = (token_dim * self.num_ns_token
                           if self.readout == "flatten" else token_dim)
        self.tower = nn.ModuleList([MLP_Block(input_dim=tower_input_dim,
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=tower_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout)
                                    for _ in range(num_tasks)])
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x S x item_info_dim

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x S x token_dim

        # target item as an NS token
        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        if self.tokenizer_uses_field_input:
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim
            )
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)           # B x num_ns_token x token_dim

        # unified OneTrans
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            unified_tokens,
            mask
        )

        # Predict from the NS tokens
        if self.readout == "mean":
            bottom_output = unified_tokens.mean(dim=1)
        else:
            bottom_output = unified_tokens.flatten(start_dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class OneTransBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 dnn_activations='ReLU',
                 expansion_factor=4,
                 attention_activation_type="SoftMax",
                 reduction_ratio=0.5):
        super(OneTransBlock, self).__init__()
        self.num_layers = num_layers
        self.reduction_ratio = float(reduction_ratio)
        # ``1.0`` disables the pyramid reduction (no sequence pruning), which is
        # used as the control group for the scaling/efficiency claim.
        if not 0.0 <= self.reduction_ratio <= 1.0:
            raise ValueError(
                "reduction_ratio must be in [0, 1], got {}"
                .format(reduction_ratio)
            )

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

        self.mha_layers = nn.ModuleList([
            MixedMHA(
                input_dim=input_dim,
                num_heads=num_heads,
                num_ns_token=num_ns_token,
                attention_activation_type=attention_activation_type,
            ) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            MixedFFN(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                dnn_activations=dnn_activations,
                expansion_factor=expansion_factor
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()
            q_mask = mask
        else:
            q_mask = None

        ps_tokens = s_tokens
        for i in range(self.num_layers):
            # Prune sequence queries only. NS tokens stay in the separate
            # ns_tokens stream and keep a fixed token count across all layers.
            num_sequence_tokens = ps_tokens.size(1)
            if self.reduction_ratio >= 1.0:
                sequence_start = 0
            else:
                sequence_start = int(num_sequence_tokens * self.reduction_ratio)
            ps_tokens = ps_tokens[:, sequence_start:, :]
            if q_mask is not None:
                q_mask = q_mask[:, sequence_start:]
            norm_s = self.norm1_layers[i](s_tokens)
            norm_ps = self.norm1_layers[i](ps_tokens)
            norm_ns = self.norm1_layers[i](ns_tokens)

            delta_ps, delta_ns = self.mha_layers[i](norm_s, norm_ps, norm_ns, kv_mask=mask, q_mask=q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            norm_ps = self.norm2_layers[i](ps_tokens)
            norm_ns = self.norm2_layers[i](ns_tokens)

            delta_ps, delta_ns = self.ffn_layers[i](norm_ps, norm_ns, q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            s_tokens = ps_tokens
            mask = q_mask
        return ns_tokens


class MixedMHA(nn.Module):
    def __init__(self, input_dim, num_heads, num_ns_token,
                 attention_activation_type="SoftMax"):
        super(MixedMHA, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads

        # shared projections for S-tokens
        self.W_q_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)

        # token-specific projections for NS-tokens
        # shape: [num_ns_token, input_dim, input_dim]
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(
            attention_activation_type=attention_activation_type
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)

    def forward(self, s_tokens, ps_tokens, ns_tokens, kv_mask=None, q_mask=None):
        """
        s_tokens:  B x Ls x D
        ps_tokens: B x qLs x D
        ns_tokens: B x Lns x D
        kv_mask:   B x Ls, valid mask for full S key/value
        q_mask:    B x qLs, valid mask for pruned S query
        Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
        Torch cannot use the causal_lower_right mask, so full attention is used here to simulate pyramid reduction.
        """
        B, Ls, D = s_tokens.shape
        _, qLs, _ = ps_tokens.shape
        _, Lns, _ = ns_tokens.shape

        if kv_mask is not None:
            s_tokens = s_tokens * kv_mask.unsqueeze(-1).float()

        # shared QKV for S-tokens
        q_s = self.W_q_s(s_tokens)    # B x Ls x D
        k_s = self.W_k_s(s_tokens)    # B x Ls x D
        v_s = self.W_v_s(s_tokens)    # B x Ls x D

        # token-specific QKV for NS-tokens
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)

        q = torch.cat([q_s, q_ns], dim=1)   # B x (Ls+Lns) x D
        k = torch.cat([k_s, k_ns], dim=1)   # B x (Ls+Lns) x D
        v = torch.cat([v_s, v_ns], dim=1)   # B x (Ls+Lns) x D

        L = Ls + Lns

        # split heads
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # B x H x L x h
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # B x H x L x h
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        output, _ = self.dot_attention(
            q,
            k,
            v,
            scale=self.head_dim ** 0.5,
            is_causal=True,
        )

        # concat heads
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_o(output)

        ps_out = output[:, Ls - qLs:Ls, :]
        ns_out = output[:, Ls:, :]

        if q_mask is not None:
            ps_out = ps_out * q_mask.unsqueeze(-1).float()

        return ps_out, ns_out


class MixedFFN(nn.Module):
    def __init__(self, input_dim, num_ns_token, dnn_activations, expansion_factor=4):
        super(MixedFFN, self).__init__()
        hidden_dim = input_dim * expansion_factor
        self.num_ns_token = num_ns_token
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = get_activation(dnn_activations)

        # shared FFN for S-tokens
        self.ffn_s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(dnn_activations),
            nn.Linear(hidden_dim, input_dim)
        )

        # token-specific FFN for NS-tokens using einsum-style parameters
        self.W1_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, hidden_dim))
        self.b1_ns = nn.Parameter(torch.zeros(num_ns_token, hidden_dim))
        self.W2_ns = nn.Parameter(torch.empty(num_ns_token, hidden_dim, input_dim))
        self.b2_ns = nn.Parameter(torch.zeros(num_ns_token, input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1_ns)
        nn.init.xavier_uniform_(self.W2_ns)
        nn.init.zeros_(self.b1_ns)
        nn.init.zeros_(self.b2_ns)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x Ls x D
        ns_tokens: B x Lns x D
        """
        # shared FFN for S-tokens
        s_out = self.ffn_s(s_tokens)

        # token-specific FFN for NS-tokens
        # layer 1: B x Lns x D  @  Lns x D x H -> B x Lns x H
        ns_hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1_ns) + self.b1_ns.unsqueeze(0)
        ns_hidden = self.activation(ns_hidden)
        # layer 2: B x Lns x H  @  Lns x H x D -> B x Lns x D
        ns_out = torch.einsum("blh,lhd->bld", ns_hidden, self.W2_ns) + self.b2_ns.unsqueeze(0)

        if mask is not None:
            s_out = s_out * mask.unsqueeze(-1).float()

        return s_out, ns_out
