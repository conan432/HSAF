import math
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from unirank.pytorch.layers.blocks import MLP_Block


class SemanticGroupTokenizer(nn.Module):
    """Tokenize ordered semantic groups without crossing group boundaries.

    Each nonzero group is assigned at least one token proportionally to its
    embedding dimension.  The group embedding is then padded to an even split
    and projected token-by-token, matching RankMixer's semantic-group + split +
    projection tokenization.  Callers must pass a flat tensor whose dimensions
    follow ``group_dims`` exactly.
    """

    def __init__(self,
                 input_dim: int,
                 token_dim: int,
                 num_tokens: int,
                 group_dims: Sequence[int],
                 activation: str = "SiLU",
                 layer_norm: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.group_dims = [int(dim) for dim in group_dims]
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if sum(self.group_dims) != self.input_dim:
            raise ValueError(
                "group_dims must sum to input_dim: "
                f"{self.group_dims} != {self.input_dim}"
            )

        nonzero_dims = [dim for dim in self.group_dims if dim > 0]
        if len(nonzero_dims) > self.num_tokens:
            raise ValueError(
                f"{len(nonzero_dims)} nonzero semantic groups require at least "
                f"num_tokens={len(nonzero_dims)}, got {self.num_tokens}"
            )

        self.token_allocations = self._allocate_tokens(self.group_dims, self.num_tokens)
        self.group_offsets = [0]
        for dim in self.group_dims:
            self.group_offsets.append(self.group_offsets[-1] + dim)

        self.token_mlps = nn.ModuleList()
        self.group_chunk_dims = []
        for group_idx, (dim, allocation) in enumerate(zip(self.group_dims, self.token_allocations)):
            if dim == 0 or allocation == 0:
                self.group_chunk_dims.append(0)
                continue
            chunk_dim = int(math.ceil(dim / allocation))
            self.group_chunk_dims.append(chunk_dim)
            for _ in range(allocation):
                self.token_mlps.append(
                    MLP_Block(
                        input_dim=chunk_dim,
                        hidden_units=[self.token_dim],
                        hidden_activations=activation,
                        layer_norm=layer_norm,
                    )
                )
        if len(self.token_mlps) != self.num_tokens:
            raise RuntimeError(
                f"Semantic tokenizer built {len(self.token_mlps)} tokens, "
                f"expected {self.num_tokens}"
            )

    @staticmethod
    def _allocate_tokens(group_dims: List[int], num_tokens: int) -> List[int]:
        nonzero = [dim > 0 for dim in group_dims]
        total_dim = sum(dim for dim, valid in zip(group_dims, nonzero) if valid)
        if total_dim == 0:
            return [num_tokens] + [0] * (len(group_dims) - 1)

        raw = [dim * num_tokens / total_dim for dim in group_dims]
        allocations = [
            max(1, int(math.floor(raw[idx]))) if nonzero[idx] else 0
            for idx in range(len(group_dims))
        ]

        # Correct minimum-token constraints using largest-deficit allocation.
        while sum(allocations) < num_tokens:
            candidates = [
                idx for idx, valid in enumerate(nonzero)
                if valid and allocations[idx] < num_tokens
            ]
            idx = max(candidates, key=lambda j: raw[j] - allocations[j])
            allocations[idx] += 1
        while sum(allocations) > num_tokens:
            candidates = [
                idx for idx, valid in enumerate(nonzero)
                if valid and allocations[idx] > 1
            ]
            if not candidates:
                raise ValueError("Cannot satisfy semantic token allocation")
            idx = min(candidates, key=lambda j: raw[j] - allocations[j])
            allocations[idx] -= 1
        return allocations

    def forward(self, feature_embeddings: torch.Tensor) -> torch.Tensor:
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
        if feature_embeddings.size(-1) != self.input_dim:
            raise ValueError(
                "SemanticGroupTokenizer expects input_dim="
                f"{self.input_dim}, got {feature_embeddings.size(-1)}"
            )

        batch_size = feature_embeddings.size(0)
        token_outputs = []
        token_index = 0
        for group_idx, (dim, allocation) in enumerate(
            zip(self.group_dims, self.token_allocations)
        ):
            if dim == 0 or allocation == 0:
                continue
            start = self.group_offsets[group_idx]
            segment = feature_embeddings[:, start:start + dim]
            pad_dim = self.group_chunk_dims[group_idx] * allocation - dim
            if pad_dim > 0:
                segment = F.pad(segment, (0, pad_dim))
            segments = segment.view(
                batch_size, allocation, self.group_chunk_dims[group_idx]
            )
            for local_index in range(allocation):
                token_outputs.append(self.token_mlps[token_index](segments[:, local_index, :]))
                token_index += 1

        if token_index != len(self.token_mlps):
            raise RuntimeError("Semantic tokenizer token traversal mismatch")
        return torch.stack(token_outputs, dim=1)
