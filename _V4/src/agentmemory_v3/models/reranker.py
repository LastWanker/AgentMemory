from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class SlotFeatureReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class LoadedReranker:
    model: SlotFeatureReranker
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: torch.device
    meta: dict[str, Any]


def save_reranker(
    path: str | Path,
    *,
    model: SlotFeatureReranker,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    hidden_dim: int,
    dropout: float,
    meta: dict[str, Any] | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_names": list(feature_names),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "meta": dict(meta or {}),
    }
    torch.save(payload, out_path)


def load_reranker(path: str | Path, *, device: str = "cpu") -> LoadedReranker | None:
    src_path = Path(path)
    if not src_path.exists():
        return None
    torch_device = torch.device(device)
    try:
        payload = torch.load(src_path, map_location=torch_device, weights_only=True)
    except Exception:
        payload = torch.load(src_path, map_location=torch_device)
    feature_names = [str(item) for item in payload["feature_names"]]
    model = SlotFeatureReranker(
        input_dim=len(feature_names),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch_device)
    model.eval()
    return LoadedReranker(
        model=model,
        feature_names=feature_names,
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
        device=torch_device,
        meta=dict(payload.get("meta") or {}),
    )


def score_feature_matrix(bundle: LoadedReranker, feature_matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=np.float32)
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = np.where(bundle.feature_std == 0.0, 1.0, bundle.feature_std)
    normalized = (features - bundle.feature_mean) / denom
    tensor = torch.from_numpy(normalized).to(bundle.device)
    with torch.no_grad():
        scores = bundle.model(tensor).detach().cpu().numpy().astype(np.float32)
    return scores
