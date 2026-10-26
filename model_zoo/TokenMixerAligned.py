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
"""TokenMixer with the mixer-family input and history-read protocol aligned.

The original TokenMixer first pools the whole behaviour sequence with target
attention and then creates ``1 + num_group_token`` interaction tokens.  That
changes both the information available to the interaction layers and the token
budget relative to the aligned RankMixer/HeMix/UniMixer family.

This variant keeps TokenMixer's own interaction operator unchanged -- the
mixer/revert path followed by per-token SwiGLU -- but aligns everything before
that operator with RankMixer:

* ``N`` dynamic queries from ``[non-sequence features || target]`` and ``N``
  learnable fixed queries read the event-level history;
* the reader is ``HeteroQueryTokenizer`` with token-specific ``W_q``;
* the resulting layout is ``[2N history tokens || N non-sequence tokens]``;
* ``N=5``, hence 15 interaction tokens;
* token width, layer count, expansion factor, tower, dropout and sequence
  window follow the RankMixer base configuration.

The legacy TokenMixer remains available and is not modified.
"""

import torch
from torch import nn

from unirank.pytorch.layers import FeatureEmbedding, HeteroQueryTokenizer, MLP_Block
from unirank.pytorch.models import MultiTaskModel

from .TokenMixer import TokenMixerBlocks


class TokenMixerAligned(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="TokenMixerAligned",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=16,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=256,
                 num_ns_token=5,
                 num_real_token=0,
                 num_heads=2,
                 real_seq_len=20,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(TokenMixerAligned, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            **kwargs
        )
        self.num_tasks = int(num_tasks)
        self.feature_map = feature_map
        self.embedding_dim = int(embedding_dim)
        self.token_dim = int(token_dim)
        self.num_ns_token = int(num_ns_token)
        self.num_real_token = int(num_real_token)
        self.real_seq_len = int(real_seq_len)
        self.accumulation_steps = int(accumulation_steps)
        if self.num_ns_token <= 0:
            raise ValueError(f"num_ns_token must be positive, got {self.num_ns_token}.")

        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels or spec.get("type") == "meta":
                continue
            emb_dim = int(spec.get("embedding_dim", self.embedding_dim))
            if spec.get("source") in ("item", "action"):
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, self.embedding_dim)

        # Exactly the mixer-family history read used by the aligned RankMixer.
        self.unified_tokenizer_layer = HeteroQueryTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=self.token_dim,
            num_ns_token=self.num_ns_token,
            num_real_token=self.num_real_token,
            num_heads=int(num_heads),
            real_seq_len=self.real_seq_len,
        )
        if self.item_info_dim != self.token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, self.token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.num_all_tokens = 3 * self.num_ns_token
        self.unified_interaction_layers = TokenMixerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_all_tokens,
            num_layers=int(num_layers),
            expand=int(expansion_factor),
            net_dropout=float(net_dropout),
        )

        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=self.unified_interaction_layers.output_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            for _ in range(self.num_tasks)
        ])
        if isinstance(task, list):
            assert len(task) == self.num_tasks, \
                "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(str(t)) for t in task]
            )
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(self.num_tasks)]
            )

        self.compile(
            kwargs.get("dense_optimizer"),
            kwargs["loss"],
            kwargs.get("dense_learning_rate"),
        )
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, :-1, :]
        sequence_tokens = self.item_token_proj(sequence_emb)

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        tokenizer_input = torch.cat([user_context_emb, target_emb], dim=-1)
        unified_tokens = self.unified_tokenizer_layer(
            tokenizer_input,
            sequence_tokens,
            mask,
        )

        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens,
        )
        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [
            self.output_activation[i](tower_output[i])
            for i in range(self.num_tasks)
        ]

        labels = self.feature_map.labels
        return {
            f"{labels[i]}_pred": y_pred[i]
            for i in range(self.num_tasks)
        }
