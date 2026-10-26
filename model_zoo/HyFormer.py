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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedAveragePooling, MultiHeadTokenMixing, PerTokenFeedForward, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer

class HyFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="HyFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 num_global_token=1,
                 expansion_factor=4,
                 sequence_encoder_type="transformer",
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(HyFormer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)


        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = num_ns_token
        self.num_global_token = num_global_token
        self.accumulation_steps = accumulation_steps
        self.masked_avg_pooling = MaskedAveragePooling()

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Non-sequential features + target item -> NS tokens.  The whole
        # unified-transformer family uses the same AutoSplit tokenizer, and
        # ``num_ns_token`` is set so that NS + global tokens match the 16-token
        # budget of OneTrans / EST.
        self.ns_tokenizer = AutoSplitTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token
        )

        # Query Generation:
        # Global Info = concat(non-seq features, target item, meanpool(sequence))
        self.query_generator = AutoSplitTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim + token_dim,
            token_dim=token_dim,
            num_tokens=num_global_token
        )

        # Project sequence and target items to token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # HyFormer backbone
        self.unified_interaction_layers = HyFormerBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=num_ns_token,
            num_global_token=num_global_token,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations,
            sequence_encoder_type=sequence_encoder_type
        )

        # Predict from the global tokens (HyFormer's native readout).  The
        # readout is intentionally **not** aligned across the family: each model
        # keeps the readout its own design defines (see the fairness checklist).
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim,
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(num_tasks)
        ])

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

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        # sequence tokens Input Tokenization
        s_tokens = self.item_token_proj(sequence_emb)  # B x T x token_dim
        s_tokens = s_tokens * mask.unsqueeze(-1).float()

        # non-seq embedding
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # B x non_item_dim

        # NS tokens Input Tokenization
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        tokens = self.ns_tokenizer(feature_embeddings)  # B x num_ns_token x token_dim

        # Global Tokens from Query Generation
        seq_pool = self.masked_avg_pooling(s_tokens, mask)  # B x token_dim
        global_info = torch.cat([user_context_emb, target_emb, seq_pool], dim=-1)
        global_tokens = self.query_generator(global_info)  # B x num_global_token x token_dim

        # HyFormer block
        _, global_tokens, ns_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            global_tokens,
            tokens,
            mask
        )

        # final prediction
        bottom_output = global_tokens.mean(dim=1)

        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class HyFormerBlock(nn.Module):
    """
    Each layer:
    1) Sequence Representation Encoding
    2) Query Decoding: global tokens -> cross attend over sequence tokens
    3) Query Boosting: concat(decoded global tokens, ns tokens) -> mixer -> residual
    """
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 num_global_token,
                 expansion_factor=4,
                 dnn_activations="ReLU",
                 sequence_encoder_type="transformer"):
        super(HyFormerBlock, self).__init__()
        self.num_layers = num_layers
        self.num_global_token = num_global_token

        self.seq_layers = nn.ModuleList([
            SequenceRepresentationLayer(
                input_dim=input_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dnn_activations=dnn_activations,
                sequence_encoder_type=sequence_encoder_type
            ) for _ in range(num_layers)
        ])

        self.decode_layers = nn.ModuleList([
            QueryDecodingLayer(
                input_dim=input_dim,
                num_heads=num_heads,
                num_query_token=num_global_token
            ) for _ in range(num_layers)
        ])

        self.boost_layers = nn.ModuleList([
            QueryBoostingLayer(
                input_dim=input_dim,
                num_global_token=num_global_token,
                num_ns_token=num_ns_token,
                expansion_factor=expansion_factor,
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, global_tokens, ns_tokens, mask=None):
        """
        s_tokens:      B x Ls x D
        global_tokens: B x G  x D
        ns_tokens:     B x M  x D
        """
        G = global_tokens.size(1)

        for i in range(self.num_layers):
            # 1) Sequence Representation Encoding
            s_tokens = self.seq_layers[i](s_tokens, mask)

            # 2) Query Decoding
            decoded_global = self.decode_layers[i](global_tokens, s_tokens, mask)

            # 3) Query Boosting
            fusion_tokens = torch.cat([decoded_global, ns_tokens], dim=1)   # B x (G+M) x D
            boosted_tokens = self.boost_layers[i](fusion_tokens)

            global_tokens = boosted_tokens[:, :G, :]
            ns_tokens = boosted_tokens[:, G:, :]

        return s_tokens, global_tokens, ns_tokens


class SequenceRepresentationLayer(nn.Module):
    """
    Corresponds to Sequence Representation Encoding in the paper.
    - Only the Full Transformer Encoding of fine-grained interactions is reproduced here, that is, standard self-attention + FFN
    - Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
    """
    def __init__(self, input_dim, num_heads, expansion_factor=4,
                 dnn_activations="ReLU", sequence_encoder_type="transformer"):
        super(SequenceRepresentationLayer, self).__init__()
        self.sequence_encoder_type = sequence_encoder_type
        self.num_heads = num_heads
        self.expansion_factor = int(expansion_factor)
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(input_dim)
        # Pre-norm before the FFN as well, matching OneTrans' norm placement.
        self.ffn_norm = nn.LayerNorm(input_dim)

        if sequence_encoder_type == "transformer":
            self.q_proj = nn.Linear(input_dim, input_dim)
            self.k_proj = nn.Linear(input_dim, input_dim)
            self.v_proj = nn.Linear(input_dim, input_dim)
            self.out_proj = nn.Linear(input_dim, input_dim)
            self.dot_attention = ScaledDotProductAttention()
        else:
            raise ValueError("sequence_encoder_type not implemented")

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * self.expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(input_dim * self.expansion_factor, input_dim)
        )

    def forward(self, s_tokens, mask=None):

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()
        x = s_tokens

        norm_x = self.norm(x)
        B, L, D = norm_x.shape

        # Q, K, V projections -> (B, num_heads, L, head_dim)
        q = self.q_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out, _ = self.dot_attention(q, k, v, scale=self.head_dim ** 0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        x = attn_out + x
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        x = self.ffn(self.ffn_norm(x)) + x
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class QueryDecodingLayer(nn.Module):
    """
    Q^(l) = CrossAttn(Q^(l-1), K^(l), V^(l))
    Pre-normalization and residual connections improve training stability.
    Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.

   """
    def __init__(self, input_dim, num_heads, num_query_token):
        super(QueryDecodingLayer, self).__init__()
        self.num_query_token = num_query_token
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(input_dim)

        # Token-specific Q over the query tokens, shared K/V/out projection:
        # the per-token scope of the whole unified-transformer family follows
        # OneTrans (token-specific projections for non-sequence tokens, shared
        # projections for the sequence side).
        self.q_proj = nn.Parameter(torch.empty(num_query_token, input_dim, input_dim))
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dot_attention = ScaledDotProductAttention()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj)

    def forward(self, global_tokens, s_tokens, mask=None):
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        norm_q = self.norm(global_tokens)
        norm_kv = self.norm(s_tokens)

        B, Lq, D = norm_q.shape
        Lk = norm_kv.shape[1]

        # Q from global tokens (token-specific weights), K/V from sequence tokens
        q_weight = self.q_proj[:Lq]
        q = torch.einsum("bld,ldh->blh", norm_q, q_weight)
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out, _ = self.dot_attention(q, k, v, scale=self.head_dim ** 0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        attn_out = self.out_proj(attn_out)

        return attn_out + global_tokens


class QueryBoostingLayer(nn.Module):
    """
    Query Boosting in the paper is actually RankMixer. You can also try to use MLP-Mixer instead of MultiHeadTokenMixing.
    """
    def __init__(self, input_dim, num_global_token, num_ns_token, expansion_factor=4):
        super(QueryBoostingLayer, self).__init__()
        self.num_tokens = num_global_token + num_ns_token
        self.input_dim = input_dim
        self.expansion_factor = int(expansion_factor)
        self.mixer_norm = nn.LayerNorm(input_dim)
        self.pffn_norm = nn.LayerNorm(input_dim)
        self.token_mixer = MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token + num_global_token)
        self.pffn = PerTokenFeedForward(input_dim=input_dim, num_token=num_ns_token + num_global_token,
                                       expand=self.expansion_factor,
                                       hidden_activation="ReLU")

    def forward(self, x: torch.Tensor):
        x = self.token_mixer(self.mixer_norm(x)) + x
        x = self.pffn(self.pffn_norm(x)) + x
        return x
