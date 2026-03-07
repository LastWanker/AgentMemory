from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class PlainSuppressorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (512, 128), dropout: float = 0.15) -> None:
        super().__init__()
        dims = [int(input_dim)] + [max(1, int(item)) for item in hidden_dims]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class OreoMemoryMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Iterable[int] = (384, 128),
        dropout: float = 0.18,
        slots_k: int = 32,
        top_r: int = 2,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        hidden = [max(1, int(item)) for item in hidden_dims]
        if not hidden:
            hidden = [256, 128]
        if len(hidden) == 1:
            hidden = [hidden[0], max(64, hidden[0] // 2)]

        self.latent_dim = int(hidden[1])
        self.slots_k = max(1, int(slots_k))
        self.top_r = max(1, int(top_r))
        self.tau = max(1e-4, float(tau))

        self.block_a = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden[0])),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(int(hidden[0]), self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )

        # Learnable key-value memory layer.
        self.memory_keys = nn.Parameter(torch.empty(self.slots_k, self.latent_dim))
        self.memory_values = nn.Parameter(torch.empty(self.slots_k, self.latent_dim))
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)

        self.block_b = nn.Sequential(
            nn.Linear(self.latent_dim * 2, int(hidden[0])),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(int(hidden[0]), 1),
        )

    def forward_logits(self, x: torch.Tensor, return_aux: bool = False):
        z = self.block_a(x)
        logits = torch.matmul(z, self.memory_keys.transpose(0, 1)) / self.tau
        if self.top_r < self.slots_k:
            top_idx = torch.topk(logits, k=min(self.top_r, self.slots_k), dim=-1).indices
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(dim=-1, index=top_idx, src=logits.gather(-1, top_idx))
        attn = torch.softmax(logits, dim=-1)
        mem = torch.matmul(attn, self.memory_values)
        final_logits = self.block_b(torch.cat([z, mem], dim=-1)).squeeze(-1)
        if return_aux:
            return final_logits, {"memory_attn": attn, "latent": z}
        return final_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class OreoTypeHeadsMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        feedback_types: Iterable[str] = ("unrelated", "toforget"),
        hidden_dims: Iterable[int] = (384, 128),
        dropout: float = 0.18,
        slots_k: int = 32,
        top_r: int = 2,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        hidden = [max(1, int(item)) for item in hidden_dims]
        if not hidden:
            hidden = [256, 128]
        if len(hidden) == 1:
            hidden = [hidden[0], max(64, hidden[0] // 2)]

        ordered_types = [str(item).strip().lower() for item in feedback_types if str(item).strip()]
        dedup_types = list(dict.fromkeys(ordered_types))
        self.feedback_types = dedup_types if dedup_types else ["unrelated", "toforget"]
        self.default_feedback_type = self.feedback_types[0]
        self.latent_dim = int(hidden[1])
        self.slots_k = max(1, int(slots_k))
        self.top_r = max(1, int(top_r))
        self.tau = max(1e-4, float(tau))

        self.block_a = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden[0])),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(int(hidden[0]), self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.memory_keys = nn.Parameter(torch.empty(self.slots_k, self.latent_dim))
        self.memory_values = nn.Parameter(torch.empty(self.slots_k, self.latent_dim))
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)
        self.heads = nn.ModuleDict(
            {
                feedback_type: nn.Sequential(
                    nn.Linear(self.latent_dim * 2, int(hidden[0])),
                    nn.GELU(),
                    nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                    nn.Linear(int(hidden[0]), 1),
                )
                for feedback_type in self.feedback_types
            }
        )

    def _encode(self, x: torch.Tensor, *, return_aux: bool = False):
        z = self.block_a(x)
        logits = torch.matmul(z, self.memory_keys.transpose(0, 1)) / self.tau
        if self.top_r < self.slots_k:
            top_idx = torch.topk(logits, k=min(self.top_r, self.slots_k), dim=-1).indices
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(dim=-1, index=top_idx, src=logits.gather(-1, top_idx))
        attn = torch.softmax(logits, dim=-1)
        mem = torch.matmul(attn, self.memory_values)
        fused = torch.cat([z, mem], dim=-1)
        if return_aux:
            return fused, {"memory_attn": attn, "latent": z}
        return fused, None

    def forward_logits(
        self,
        x: torch.Tensor,
        *,
        feedback_type: str | None = None,
        feedback_types: list[str] | tuple[str, ...] | None = None,
        return_aux: bool = False,
    ):
        fused, aux = self._encode(x, return_aux=return_aux)
        if feedback_types is not None:
            batch_types = [str(item or "").strip().lower() for item in feedback_types]
            if len(batch_types) != int(fused.shape[0]):
                raise ValueError("feedback_types length must match batch size")
            out = fused.new_zeros((int(fused.shape[0]),))
            for feedback_name in set(batch_types):
                head_name = feedback_name if feedback_name in self.heads else self.default_feedback_type
                indices = [idx for idx, value in enumerate(batch_types) if value == feedback_name]
                idx_tensor = torch.as_tensor(indices, device=fused.device, dtype=torch.long)
                out.index_copy_(0, idx_tensor, self.heads[head_name](fused.index_select(0, idx_tensor)).squeeze(-1))
            if return_aux:
                return out, aux or {}
            return out
        head_name = str(feedback_type or self.default_feedback_type).strip().lower()
        if head_name not in self.heads:
            head_name = self.default_feedback_type
        logits = self.heads[head_name](fused).squeeze(-1)
        if return_aux:
            return logits, aux or {}
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))
