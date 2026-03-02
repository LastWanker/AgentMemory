"""Bipartite alignment scorer with a small set-transformer style encoder."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import Vector


def vector_group_to_fixed(
    vecs: List[Vector],
    *,
    size: int,
    target_dim: int,
) -> Tuple[Tensor, Tensor]:
    fixed = torch.zeros((size, target_dim), dtype=torch.float32)
    mask = torch.zeros((size,), dtype=torch.bool)
    rows = min(size, len(vecs))
    for row_idx in range(rows):
        raw = vecs[row_idx]
        if not raw:
            continue
        vec = torch.tensor(raw, dtype=torch.float32)
        cols = min(target_dim, int(vec.numel()))
        if cols <= 0:
            continue
        fixed[row_idx, :cols] = vec[:cols]
        mask[row_idx] = True
    return fixed, mask


class BipartiteAlignTransformer(nn.Module):
    """Soft alignment + set transformer style scorer."""

    def __init__(
        self,
        *,
        input_dim: int = 384,
        seq_len: int = 8,
        proj_dim: int = 192,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        tau: float = 0.1,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.proj_dim = int(proj_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learnable_tau = bool(learnable_tau)

        self.shared_proj = nn.Linear(self.input_dim, self.proj_dim)
        self.token_in = nn.Linear(self.proj_dim * 4, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1),
        )

        if self.learnable_tau:
            self.tau_logit = nn.Parameter(torch.log(torch.tensor(float(tau), dtype=torch.float32)))
        else:
            self.register_buffer("tau_const", torch.tensor(float(tau), dtype=torch.float32))

    def current_tau(self) -> Tensor:
        if self.learnable_tau:
            return torch.exp(self.tau_logit).clamp_min(1e-4)
        return self.tau_const.clamp_min(1e-4)

    def forward(
        self,
        q_batch: Tensor,
        m_batch: Tensor,
        *,
        q_mask: Optional[Tensor] = None,
        m_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # q_batch / m_batch: [B, L, D]
        q = F.normalize(q_batch, p=2, dim=-1, eps=1e-6)
        m = F.normalize(m_batch, p=2, dim=-1, eps=1e-6)

        q_proj = F.normalize(self.shared_proj(q), p=2, dim=-1, eps=1e-6)
        m_proj = F.normalize(self.shared_proj(m), p=2, dim=-1, eps=1e-6)

        sim = torch.matmul(q_proj, m_proj.transpose(1, 2)) / self.current_tau()
        if m_mask is not None:
            m_valid = m_mask.unsqueeze(1).bool()
            sim = sim.masked_fill(~m_valid, -1e4)

        align = torch.softmax(sim, dim=-1)
        if m_mask is not None:
            align = align * m_mask.unsqueeze(1).float()
            align = align / align.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        m_aligned = torch.matmul(align, m_proj)

        tokens = torch.cat(
            [q_proj, m_aligned, q_proj - m_aligned, q_proj * m_aligned],
            dim=-1,
        )
        hidden = self.token_in(tokens)

        src_key_padding_mask = None
        if q_mask is not None:
            src_key_padding_mask = ~q_mask.bool()
        encoded = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)

        if q_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = q_mask.unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class BipartiteLearnedFieldScorer:
    """Inference scorer for bipartite align model."""

    def __init__(
        self,
        reranker_path: str,
        device: str = "cpu",
        batch_size: int = 512,
        mem_cache_limit: int = 20000,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.mem_cache_limit = max(0, int(mem_cache_limit))
        self._mem_fixed_cache: Dict[int, Tuple[Tensor, Tensor]] = {}
        self.meta: Dict[str, object] = {}

        try:
            state = torch.load(reranker_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(reranker_path, map_location=self.device)

        if isinstance(state, dict):
            loaded_meta = state.get("meta", {})
            if isinstance(loaded_meta, dict):
                self.meta = dict(loaded_meta)

        model_cfg = {
            "input_dim": int(self.meta.get("bipartite_input_dim", 384)),
            "seq_len": int(self.meta.get("bipartite_seq_len", 8)),
            "proj_dim": int(self.meta.get("bipartite_proj_dim", 192)),
            "d_model": int(self.meta.get("bipartite_d_model", 256)),
            "num_heads": int(self.meta.get("bipartite_num_heads", 4)),
            "num_layers": int(self.meta.get("bipartite_num_layers", 3)),
            "dropout": float(self.meta.get("bipartite_dropout", 0.1)),
            "tau": float(self.meta.get("bipartite_tau", 0.1)),
            "learnable_tau": bool(self.meta.get("bipartite_learnable_tau", False)),
        }
        self.model = BipartiteAlignTransformer(**model_cfg).to(self.device)
        state_dict = state.get("model_state", state)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.input_dim = int(self.model.input_dim)
        self.seq_len = int(self.model.seq_len)

    def _to_fixed(self, vecs: List[Vector]) -> Tuple[Tensor, Tensor]:
        return vector_group_to_fixed(
            vecs,
            size=self.seq_len,
            target_dim=self.input_dim,
        )

    def _memory_fixed(self, m_vecs: List[Vector]) -> Tuple[Tensor, Tensor]:
        cache_key = id(m_vecs)
        cached = self._mem_fixed_cache.get(cache_key)
        if cached is not None:
            return cached
        fixed = self._to_fixed(m_vecs)
        if self.mem_cache_limit <= 0 or len(self._mem_fixed_cache) < self.mem_cache_limit:
            self._mem_fixed_cache[cache_key] = fixed
        return fixed

    def _semantic_from_raw(self, raw_score: float) -> float:
        # Keep raw logit as ranking score to avoid sigmoid compression.
        return float(raw_score)

    def score(self, q_vecs: List[Vector], m_vecs: List[Vector]) -> Tuple[float, Dict[str, List[float]]]:
        if not q_vecs or not m_vecs:
            empty = {"raw_score": [0.0], "semantic_score": [0.0], "tau": [float(self.model.current_tau().item())]}
            return 0.0, empty
        with torch.no_grad():
            q_fixed, q_mask = self._to_fixed(q_vecs)
            m_fixed, m_mask = self._memory_fixed(m_vecs)
            q_batch = q_fixed.unsqueeze(0).to(self.device)
            m_batch = m_fixed.unsqueeze(0).to(self.device)
            q_mask_batch = q_mask.unsqueeze(0).to(self.device)
            m_mask_batch = m_mask.unsqueeze(0).to(self.device)
            raw_tensor = self.model(
                q_batch,
                m_batch,
                q_mask=q_mask_batch,
                m_mask=m_mask_batch,
            )
            raw_score = float(raw_tensor.squeeze(0).item())
            semantic_score = self._semantic_from_raw(raw_score)
            tau_value = float(self.model.current_tau().item())
        debug = {
            "raw_score": [raw_score],
            "semantic_score": [semantic_score],
            "tau": [tau_value],
        }
        return semantic_score, debug

    def score_many(
        self,
        q_vecs: List[Vector],
        mem_vecs_list: List[List[Vector]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, List[float]]]]:
        if not mem_vecs_list:
            return []
        if not q_vecs:
            return [(0.0, {"raw_score": [0.0], "semantic_score": [0.0]}) for _ in mem_vecs_list]
        outputs: List[Tuple[float, Dict[str, List[float]]]] = []
        effective_batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        q_fixed, q_mask = self._to_fixed(q_vecs)
        tau_value = float(self.model.current_tau().item())

        with torch.no_grad():
            for start in range(0, len(mem_vecs_list), effective_batch_size):
                chunk = mem_vecs_list[start : start + effective_batch_size]
                fixed_pairs = [self._memory_fixed(m_vecs) for m_vecs in chunk]
                mem_batch = torch.stack([pair[0] for pair in fixed_pairs], dim=0)
                mem_mask = torch.stack([pair[1] for pair in fixed_pairs], dim=0)
                q_batch = q_fixed.unsqueeze(0).expand(len(chunk), -1, -1)
                q_mask_batch = q_mask.unsqueeze(0).expand(len(chunk), -1)

                if self.device.type != "cpu":
                    q_batch = q_batch.to(self.device, non_blocking=True)
                    q_mask_batch = q_mask_batch.to(self.device, non_blocking=True)
                    mem_batch = mem_batch.to(self.device, non_blocking=True)
                    mem_mask = mem_mask.to(self.device, non_blocking=True)
                else:
                    q_batch = q_batch.to(self.device)
                    q_mask_batch = q_mask_batch.to(self.device)
                    mem_batch = mem_batch.to(self.device)
                    mem_mask = mem_mask.to(self.device)

                raw_tensor = self.model(
                    q_batch,
                    mem_batch,
                    q_mask=q_mask_batch,
                    m_mask=mem_mask,
                ).detach().cpu()
                raw_scores = raw_tensor.tolist()
                for raw_score in raw_scores:
                    semantic_score = self._semantic_from_raw(float(raw_score))
                    outputs.append(
                        (
                            semantic_score,
                            {
                                "raw_score": [float(raw_score)],
                                "semantic_score": [semantic_score],
                                "tau": [tau_value],
                            },
                        )
                    )
        return outputs

