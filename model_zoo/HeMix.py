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
from unirank.pytorch.layers import (
    FeatureEmbedding,
    MLP_Block,
    HeteroQueryTokenizer,
    HeteroMixer,
)


class HeMix(MultiTaskModel):
    """HeMix: mixture baseline whose innovation is the long/short-term history read.

    The whole mixer family (RankMixer / UniMixer / HeMix) shares the same history
    reading recipe from :class:`HeteroQueryTokenizer`: ``N`` dynamic queries
    (non-sequence tokens built from user context + target) and ``N`` learnable
    fixed queries read the behaviour sequence and produce ``2N`` history tokens,
    giving ``3N`` interaction tokens in total.

    HeMix keeps its own contribution -- a two-window read: the most recent
    ``real_seq_len`` items are read by the ``num_real_token`` "real" queries and
    the earlier items by the remaining "glocal" queries.  RankMixer and UniMixer
    use a single window (``num_real_token = 0``) and are otherwise identical here.
    """

    def __init__(self,
                 feature_map,
                 model_id="HeMix",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=5,
                 num_real_token=1,
                 num_heads=2,
                 num_mix_heads=2,
                 low_rank_dim=16,
                 net_dropout=0,
                 accumulation_steps=1,
                 real_seq_len=20,
                 **kwargs):
        super(HeMix, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    **kwargs)

        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = int(num_ns_token)
        self.num_real_token = int(num_real_token)
        self.accumulation_steps = accumulation_steps
        self.real_seq_len = real_seq_len

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

        # Non-sequence features + target item -> N dynamic queries; together with
        # N learnable fixed queries they read the history into 2N history tokens.
        tokenizer_input_dim = self.non_item_dim + self.item_info_dim
        self.unified_tokenizer_layer = HeteroQueryTokenizer(
            input_dim=tokenizer_input_dim,
            token_dim=token_dim,
            num_ns_token=self.num_ns_token,
            num_real_token=self.num_real_token,
            num_heads=num_heads,
            real_seq_len=real_seq_len,
        )

        # Project sequence and target items to token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # Total number of interaction tokens = 2N history tokens + N NS tokens.
        self.num_all_tokens = 3 * self.num_ns_token
        self.unified_interaction_layers = HeteroMixer(
            input_dim=token_dim,
            num_token=self.num_all_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            num_mix_heads=num_mix_heads,
            low_rank_dim=low_rank_dim,
            net_dropout=net_dropout
        )

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

        if mask is not None:
            batch_size = mask.shape[0]
        else:
            any_tensor = next(iter(item_dict.values()))
            batch_size = any_tensor.shape[0]

        # item_dict: [history_items..., target_item]
        # reshape flatten_emb=True into B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, :-1, :]   # B x S x item_info_dim
        sequence_tokens = self.item_token_proj(sequence_emb)   # B x S x token_dim

        # Non-sequential features + target item -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)   # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)

        # tokens: [2N history tokens, N NS tokens]
        unified_tokens = self.unified_tokenizer_layer(
            feature_embeddings,
            sequence_tokens,
            mask
        )

        # HeteroMixer interaction
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens
        )

        # prediction
        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
