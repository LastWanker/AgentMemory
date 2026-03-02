from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SlotBipartiteAlignTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 192,
        seq_len: int = 8,
        proj_dim: int = 128,
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
        q_mask: Tensor | None = None,
        m_mask: Tensor | None = None,
    ) -> Tensor:
        q = F.normalize(q_batch, p=2, dim=-1, eps=1e-6)
        m = F.normalize(m_batch, p=2, dim=-1, eps=1e-6)
        q_proj = F.normalize(self.shared_proj(q), p=2, dim=-1, eps=1e-6)
        m_proj = F.normalize(self.shared_proj(m), p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(q_proj, m_proj.transpose(1, 2)) / self.current_tau()
        if m_mask is not None:
            sim = sim.masked_fill(~m_mask.unsqueeze(1).bool(), -1e4)
        align = torch.softmax(sim, dim=-1)
        if m_mask is not None:
            align = align * m_mask.unsqueeze(1).float()
            align = align / align.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        m_aligned = torch.matmul(align, m_proj)
        tokens = torch.cat([q_proj, m_aligned, q_proj - m_aligned, q_proj * m_aligned], dim=-1)
        hidden = self.token_in(tokens)
        src_key_padding_mask = ~q_mask.bool() if q_mask is not None else None
        encoded = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)
        if q_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = q_mask.unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


@dataclass
class LoadedSlotBipartite:
    model: SlotBipartiteAlignTransformer
    device: torch.device
    meta: dict[str, Any]


def save_slot_bipartite(path: str | Path, *, model: SlotBipartiteAlignTransformer, meta: dict[str, Any] | None = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": "slot_bipartite_8x8",
        "model_state": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, out_path)


def load_slot_bipartite(path: str | Path, *, device: str = "cpu") -> LoadedSlotBipartite | None:
    src_path = Path(path)
    if not src_path.exists():
        return None
    torch_device = torch.device(device)
    try:
        payload = torch.load(src_path, map_location=torch_device, weights_only=True)
    except TypeError:
        payload = torch.load(src_path, map_location=torch_device)
    meta = dict(payload.get("meta") or {})
    model = SlotBipartiteAlignTransformer(
        input_dim=int(meta.get("input_dim", 192)),
        seq_len=int(meta.get("seq_len", 8)),
        proj_dim=int(meta.get("proj_dim", 128)),
        d_model=int(meta.get("d_model", 256)),
        num_heads=int(meta.get("num_heads", 4)),
        num_layers=int(meta.get("num_layers", 3)),
        dropout=float(meta.get("dropout", 0.1)),
        tau=float(meta.get("tau", 0.1)),
        learnable_tau=bool(meta.get("learnable_tau", False)),
    )
    model.load_state_dict(payload.get("model_state", payload), strict=False)
    model.to(torch_device)
    model.eval()
    return LoadedSlotBipartite(model=model, device=torch_device, meta=meta)


def score_many_slot_bipartite(
    bundle: LoadedSlotBipartite,
    *,
    q_seq: np.ndarray,
    q_mask: np.ndarray,
    mem_seqs: np.ndarray,
    mem_masks: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    if mem_seqs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q_seq_np = np.asarray(q_seq, dtype=np.float32)
    q_mask_np = np.asarray(q_mask, dtype=bool)
    mem_seq_np = np.asarray(mem_seqs, dtype=np.float32)
    mem_mask_np = np.asarray(mem_masks, dtype=bool)
    outputs: list[np.ndarray] = []
    for start in range(0, mem_seq_np.shape[0], max(1, int(batch_size))):
        chunk_mem = mem_seq_np[start : start + batch_size]
        chunk_mask = mem_mask_np[start : start + batch_size]
        q_batch = np.repeat(q_seq_np[None, :, :], chunk_mem.shape[0], axis=0)
        q_mask_batch = np.repeat(q_mask_np[None, :], chunk_mem.shape[0], axis=0)
        with torch.no_grad():
            q_tensor = torch.from_numpy(q_batch).to(bundle.device)
            q_mask_tensor = torch.from_numpy(q_mask_batch).to(bundle.device)
            m_tensor = torch.from_numpy(chunk_mem).to(bundle.device)
            m_mask_tensor = torch.from_numpy(chunk_mask).to(bundle.device)
            scores = bundle.model(q_tensor, m_tensor, q_mask=q_mask_tensor, m_mask=m_mask_tensor)
            outputs.append(scores.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)
