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

import math
import torch
from torch import nn
import torch.nn.functional as F
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, PerTokenSwiGLU
from unirank.pytorch.layers import HeteroQueryTokenizer
from unirank.pytorch.torch_utils import disable_torch_compile


class UniMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="UniMixer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=256,
                 num_ns_token=None,
                 num_real_token=0,
                 num_heads=2,
                 real_seq_len=20,
                 num_tokens=None,
                 num_basis=4,
                 global_rank=64,
                 sinkhorn_iter=5,
                 sinkhorn_eps=1e-6,
                 tau=1.0,
                 tau_end=0.05,
                 tau_schedule="linear",
                 tau_anneal_steps=20000,
                 tau_warmup_steps=0,
                 use_out_proj=False,
                 use_local_coef_softmax=False,
                 use_tau_symmetry=True,
                 net_dropout=0,
                 accumulation_steps=1,
                 max_len=100,
                 **kwargs):
        super(UniMixer, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            **kwargs
        )
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.max_len = int(max_len)
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}.")
        self.user_context_dim = 0
        self.item_info_dim = 0
        self.user_context_feature_fields = 0
        self.item_feature_fields = 0

        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue

            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.item_feature_fields += 1
            else:
                self.user_context_dim += emb_dim
                self.user_context_feature_fields += 1

        # ``num_tokens`` is the legacy alias of ``num_ns_token``.
        if num_ns_token is None:
            num_ns_token = num_tokens if num_tokens is not None else 5
        self.num_ns_token = int(num_ns_token)
        self.num_real_token = int(num_real_token)
        if self.num_ns_token <= 0:
            raise ValueError(f"num_ns_token must be positive, got {self.num_ns_token}.")

        self.num_tokens = 3 * self.num_ns_token
        self.tokenizer_input_dim = self.user_context_dim + self.item_info_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Mixer-family history read: N dynamic queries (user context + target) and
        # N learnable fixed queries read the sequence into 2N history tokens.
        self.unified_tokenizer_layer = HeteroQueryTokenizer(
            input_dim=self.tokenizer_input_dim,
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

        self.unified_interaction_layers = UniMixingLiteBlocks(
            token_dim=token_dim,
            num_token=self.num_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            num_basis=num_basis,
            global_rank=global_rank,
            sinkhorn_iter=sinkhorn_iter,
            sinkhorn_eps=sinkhorn_eps,
            tau_start=tau,
            tau_end=tau_end,
            tau_schedule=tau_schedule,
            tau_anneal_steps=tau_anneal_steps,
            tau_warmup_steps=tau_warmup_steps,
            use_out_proj=use_out_proj,
            use_local_coef_softmax=use_local_coef_softmax,
            use_tau_symmetry=use_tau_symmetry,
            net_dropout=net_dropout
        )

        tower_input_dim = token_dim

        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=tower_input_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
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

    def get_current_tau(self):
        return self.unified_interaction_layers.get_current_tau()

    @disable_torch_compile
    def update_tau(self, step=None):
        if step is None:
            step = getattr(self, "_total_steps", 0)
        return self.unified_interaction_layers.update_tau(step)

    def train_step(self, batch_data):
        """Update the Sinkhorn temperature before each training forward pass."""
        self.update_tau(step=self._total_steps)
        return super().train_step(batch_data)

    def _sync_tau_before_eval(self):
        """Use one shared temperature when blocked DDP ranks have uneven batches."""
        if self._is_distributed():
            total_steps = torch.tensor([self._total_steps], dtype=torch.int64, device=self.device)
            torch.distributed.all_reduce(total_steps, op=torch.distributed.ReduceOp.MAX)
            self._total_steps = int(total_steps.item())
        self.update_tau(step=self._total_steps)

    def eval_step(self):
        self._sync_tau_before_eval()
        return super().eval_step()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]        # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]    # B x T x item_info_dim
        sequence_tokens = self.item_token_proj(sequence_emb)   # B x T x token_dim

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)

        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)

        # tokens: [2N history tokens, N NS tokens]
        unified_tokens = self.unified_tokenizer_layer(
            feature_embeddings,
            sequence_tokens,
            mask
        )

        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict[f"{labels[i]}_pred"] = y_pred[i]
        return return_dict


class UniMixingLiteBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_layers,
                 expand=2,
                 num_basis=4,
                 global_rank=4,
                 sinkhorn_iter=5,
                 sinkhorn_eps=1e-6,
                 tau_start=1.0,
                 tau_end=None,
                 tau_schedule="linear",
                 tau_anneal_steps=None,
                 tau_warmup_steps=0,
                 use_out_proj=False,
                 use_local_coef_softmax=False,
                 use_tau_symmetry=True,
                 net_dropout=0.0):
        super(UniMixingLiteBlocks, self).__init__()
        self.num_token = int(num_token)
        self.token_dim = int(token_dim)
        self.num_layers = int(num_layers)

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_start if tau_end is None else tau_end)
        self.tau_schedule = str(tau_schedule).lower()
        self.tau_anneal_steps = None if tau_anneal_steps is None else int(tau_anneal_steps)
        self.tau_warmup_steps = int(tau_warmup_steps)

        if self.tau_start <= 0 or self.tau_end <= 0:
            raise ValueError("tau_start and tau_end must be positive.")

        if self.tau_schedule not in ["cosine", "linear", "exp"]:
            raise ValueError("tau_schedule must be one of ['cosine', 'linear', 'exp'].")

        self.register_buffer("current_tau", torch.tensor(self.tau_start, dtype=torch.float32))

        self.mixers = nn.ModuleList([
            UniMixingLiteLayer(
                token_dim=token_dim,
                num_token=num_token,
                num_basis=num_basis,
                global_rank=global_rank,
                sinkhorn_iter=sinkhorn_iter,
                sinkhorn_eps=sinkhorn_eps,
                tau=tau_start,
                use_out_proj=use_out_proj,
                use_local_coef_softmax=use_local_coef_softmax,
                use_tau_symmetry=use_tau_symmetry,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

        self.pffns = nn.ModuleList([
            SiameseFeedForwardLayer(
                token_dim=token_dim,
                num_token=num_token,
                expand=expand,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(token_dim)

        self.update_tau(step=0)

    def _compute_tau(self, step: int):
        step = max(0, int(step))

        if self.tau_anneal_steps is None or self.tau_anneal_steps <= 0:
            return self.tau_start

        if step <= self.tau_warmup_steps:
            return self.tau_start

        progress = (step - self.tau_warmup_steps) / max(1, self.tau_anneal_steps)
        progress = min(max(progress, 0.0), 1.0)

        if self.tau_schedule == "linear":
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
        elif self.tau_schedule == "cosine":
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1.0 + math.cos(math.pi * progress))
        elif self.tau_schedule == "exp":
            tau = self.tau_start * ((self.tau_end / self.tau_start) ** progress)
        else:
            raise ValueError(f"Unsupported tau_schedule={self.tau_schedule}")

        return max(float(tau), 1e-6)

    def update_tau(self, step: int):
        tau = self._compute_tau(step)
        self.current_tau.fill_(tau)
        for mixer in self.mixers:
            mixer.set_tau(tau)
        return tau

    def get_current_tau(self):
        return float(self.current_tau.item())

    def forward(self, x: torch.Tensor):
        x_stream = x
        y_stream = x

        # Engineering-stable variant: apply Siamese updates once for mixing and
        # once for the per-token FFN in each stacked layer.
        for mixer, pffn in zip(self.mixers, self.pffns):
            x_stream, y_stream = mixer(x_stream, y_stream)
            x_stream, y_stream = pffn(x_stream, y_stream)

        return x_stream + self.final_norm(y_stream)


class UniMixingLiteLayer(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_basis=4,
                 global_rank=4,
                 sinkhorn_iter=5,
                 sinkhorn_eps=1e-6,
                 tau=1.0,
                 use_out_proj=False,
                 use_local_coef_softmax=False,
                 use_tau_symmetry=True,
                 net_dropout=0.0):
        super(UniMixingLiteLayer, self).__init__()

        self.token_dim = int(token_dim)
        self.num_token = int(num_token)
        self.num_basis = int(num_basis)
        self.global_rank = int(global_rank)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.use_out_proj = bool(use_out_proj)
        self.use_local_coef_softmax = bool(use_local_coef_softmax)
        self.use_tau_symmetry = bool(use_tau_symmetry)

        self.register_buffer("current_tau", torch.tensor(float(tau), dtype=torch.float32))

        self.local_basis = nn.Parameter(
            torch.empty(self.num_basis, self.token_dim, self.token_dim)
        )
        self.local_coef = nn.Parameter(
            torch.empty(self.num_token, self.num_basis)
        )

        self.global_left = nn.Parameter(
            torch.empty(self.num_token, self.global_rank)
        )
        self.global_right = nn.Parameter(
            torch.empty(self.global_rank, self.num_token)
        )

        self.norm_x = nn.LayerNorm(self.token_dim)
        self.norm_y = nn.LayerNorm(self.token_dim)

        self.out_proj = nn.Linear(self.token_dim, self.token_dim, bias=False) if use_out_proj else nn.Identity()
        self.dropout = nn.Dropout(net_dropout)

        self.reset_parameters()

    def set_tau(self, tau: float):
        self.current_tau.fill_(max(float(tau), 1e-6))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.local_basis)
        nn.init.xavier_uniform_(self.local_coef)
        nn.init.xavier_uniform_(self.global_left)
        nn.init.xavier_uniform_(self.global_right)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)

    def sinkhorn(self, mat: torch.Tensor):
        for _ in range(self.sinkhorn_iter):
            mat = mat / (mat.sum(dim=-1, keepdim=True) + self.sinkhorn_eps)
            mat = mat / (mat.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
        return mat

    def apply_sinkhorn_constraint(self, logits: torch.Tensor):
        # Sinkhorn's exp/normalization is sensitive to low precision and is always calculated in FP32.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            tau = self.current_tau.to(device=logits.device, dtype=torch.float32).clamp_min(1e-6)
            logits = logits / tau
            logits = logits - torch.amax(logits, dim=(-1, -2), keepdim=True)
            mat = torch.exp(logits)
            return self.sinkhorn(mat)

    def local_mixing(self, z: torch.Tensor):
        coef = self.local_coef
        if self.use_local_coef_softmax:
            coef = F.softmax(coef, dim=-1)

        W_local_logits = torch.einsum("mr,rij->mij", coef, self.local_basis)
        if self.use_tau_symmetry:
            W_local_logits = 0.5 * (W_local_logits + W_local_logits.transpose(-1, -2))

        W_local = self.apply_sinkhorn_constraint(W_local_logits)
        return torch.einsum("bmi,mij->bmj", z, W_local)

    def global_mixing(self, z_local: torch.Tensor):
        G_logits = torch.matmul(self.global_left, self.global_right)
        G_logits = G_logits / math.sqrt(self.global_rank)
        if self.use_tau_symmetry:
            G_logits = 0.5 * (G_logits + G_logits.transpose(-1, -2))

        G = self.apply_sinkhorn_constraint(G_logits)
        return torch.einsum("mn,bnj->bmj", G, z_local)

    def mixing_transform(self, x: torch.Tensor):
        batch_size, num_token, token_dim = x.shape
        if num_token != self.num_token or token_dim != self.token_dim:
            raise ValueError(
                f"UniMixingLiteLayer expects [B, {self.num_token}, {self.token_dim}], "
                f"got [B, {num_token}, {token_dim}]."
            )

        z_local = self.local_mixing(x)
        z_global = self.global_mixing(z_local)
        y = self.out_proj(z_global)
        return self.dropout(y)

    def siamese_norm(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        y_norm = self.norm_y(y_stream)
        fused_input = x_stream + y_norm
        update = self.mixing_transform(fused_input)
        x_next = self.norm_x(x_stream + update)
        y_next = y_stream + update
        return x_next, y_next

    def forward(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        return self.siamese_norm(x_stream, y_stream)


class SiameseFeedForwardLayer(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 expand=2,
                 net_dropout=0.0):
        super(SiameseFeedForwardLayer, self).__init__()

        self.norm_x = nn.LayerNorm(token_dim)
        self.norm_y = nn.LayerNorm(token_dim)

        self.pffn = PerTokenSwiGLU(
            input_dim=token_dim,
            num_token=num_token,
            expand=expand,
            net_dropout=net_dropout
        )

    def siamese_norm(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        y_norm = self.norm_y(y_stream)
        fused_input = x_stream + y_norm
        update = self.pffn(fused_input)
        x_next = self.norm_x(x_stream + update)
        y_next = y_stream + update
        return x_next, y_next

    def forward(self, x_stream: torch.Tensor, y_stream: torch.Tensor):
        return self.siamese_norm(x_stream, y_stream)
