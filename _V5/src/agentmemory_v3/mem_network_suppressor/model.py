from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LegacyMLPScorer(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dims: Iterable[int] = (640, 192), dropout: float = 0.15) -> None:
        super().__init__()
        hidden = [max(1, int(item)) for item in hidden_dims]
        if not hidden:
            hidden = [640, 192]
        dims = [max(1, int(input_dim))] + hidden
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


class MemNetworkScorer(nn.Module):
    """Memory-network scorer with configurable output heads."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        state_dim: int = 256,
        num_hops: int = 2,
        num_heads: int = 1,
        memory_id_vocab_size: int = 2,
        type_vocab_size: int = 2,
        lane_vocab_size: int = 3,
        type_emb_dim: int = 16,
        lane_emb_dim: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.state_dim = max(32, int(state_dim))
        self.num_hops = max(1, int(num_hops))
        self.num_heads = max(1, int(num_heads))
        self.memory_id_emb = nn.Embedding(max(2, int(memory_id_vocab_size)), self.state_dim)

        self.type_emb = nn.Embedding(max(2, int(type_vocab_size)), max(4, int(type_emb_dim)))
        self.lane_emb = nn.Embedding(max(3, int(lane_vocab_size)), max(4, int(lane_emb_dim)))

        state_in_dim = self.embedding_dim * 4
        slot_in_dim = self.embedding_dim * 2 + self.type_emb.embedding_dim + self.lane_emb.embedding_dim + self.state_dim
        self.state_init = nn.Sequential(
            nn.Linear(state_in_dim, self.state_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.candidate_id_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.key_proj = nn.Linear(slot_in_dim, self.state_dim)
        self.value_proj = nn.Linear(slot_in_dim, self.state_dim)
        self.hop_update = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.output = nn.Sequential(
            nn.Linear(self.state_dim * 4 + 2, self.state_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.state_dim, self.num_heads),
        )

    def forward_logits(
        self,
        *,
        query_vec: torch.Tensor,
        candidate_vec: torch.Tensor,
        feedback_query_slots: torch.Tensor,
        feedback_memory_slots: torch.Tensor,
        feedback_type_ids: torch.Tensor,
        feedback_lane_ids: torch.Tensor,
        candidate_memory_id_ids: torch.Tensor | None = None,
        feedback_memory_id_ids: torch.Tensor | None = None,
        feedback_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = query_vec.float()
        c = candidate_vec.float()
        fq = feedback_query_slots.float()
        fm = feedback_memory_slots.float()
        ft = feedback_type_ids.long()
        fl = feedback_lane_ids.long()
        batch = int(q.shape[0])
        slot_count = int(fq.shape[1]) if fq.ndim >= 2 else 0
        if candidate_memory_id_ids is None:
            candidate_memory_id_ids = torch.zeros((batch,), dtype=torch.long, device=q.device)
        if feedback_memory_id_ids is None:
            feedback_memory_id_ids = torch.zeros((batch, slot_count), dtype=torch.long, device=q.device)
        cid = candidate_memory_id_ids.long().to(q.device)
        sid = feedback_memory_id_ids.long().to(q.device)
        max_idx = int(self.memory_id_emb.num_embeddings - 1)
        cid = torch.clamp(cid, min=0, max=max_idx)
        sid = torch.clamp(sid, min=0, max=max_idx)
        cid_emb = self.memory_id_emb(cid)

        # [B, 4D]
        state0 = self.state_init(torch.cat([q, c, torch.abs(q - c), q * c], dim=-1))
        state0 = state0 + self.candidate_id_proj(cid_emb)

        # [B, K, 2D + Et + El + Emid]
        slot_repr = torch.cat([fq, fm, self.type_emb(ft), self.lane_emb(fl), self.memory_id_emb(sid)], dim=-1)
        keys = self.key_proj(slot_repr)
        values = self.value_proj(slot_repr)

        if feedback_mask is None:
            mask = torch.ones(keys.shape[:2], dtype=torch.float32, device=keys.device)
        else:
            mask = feedback_mask.float().to(keys.device)

        state = state0
        last_ctx = torch.zeros_like(state0)
        last_attn = torch.zeros(mask.shape, dtype=torch.float32, device=keys.device)
        inv_temp = 1.0 / math.sqrt(float(max(1, self.state_dim)))
        for _ in range(self.num_hops):
            scores = torch.sum(keys * state.unsqueeze(1), dim=-1) * inv_temp
            scores = scores.masked_fill(mask <= 0.0, -1e4)
            attn = torch.softmax(scores, dim=-1)
            attn = attn * mask
            attn = attn / torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-6)
            ctx = torch.sum(attn.unsqueeze(-1) * values, dim=1)
            state = self.hop_update(torch.cat([state, ctx], dim=-1))
            last_ctx = ctx
            last_attn = attn

        id_match = (sid == cid.unsqueeze(1)).float() * mask
        id_hit = torch.max(id_match, dim=-1).values.unsqueeze(-1)
        id_attn_mass = torch.sum(last_attn * id_match, dim=-1, keepdim=True)
        logits = self.output(torch.cat([state, last_ctx, state0, cid_emb, id_hit, id_attn_mass], dim=-1))
        return logits, last_attn

    def forward(self, **kwargs) -> torch.Tensor:
        logits, _ = self.forward_logits(**kwargs)
        return torch.sigmoid(logits)
