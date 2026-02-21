"""PyTorch 版 tiny learned scorer。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .utils import Vector


def compute_sim_matrix(q_vecs: List[Vector], m_vecs: List[Vector], size: int = 8) -> Tensor:
    """计算并返回固定 shape=(8, 8) 的相似度矩阵。"""

    matrix = torch.zeros((size, size), dtype=torch.float32)
    if not q_vecs or not m_vecs:
        return matrix

    q_tensor = torch.tensor(q_vecs[:size], dtype=torch.float32)
    m_tensor = torch.tensor(m_vecs[:size], dtype=torch.float32)
    sim = q_tensor @ m_tensor.t()
    matrix[: sim.shape[0], : sim.shape[1]] = sim
    return matrix


class TinyReranker(nn.Module):
    """置换鲁棒 tiny reranker：统计池化 + 两层 MLP。"""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        input_dim = 8 + 8 + 8 + 8 + 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(self, sim_matrix: Tensor) -> Tensor:
        if sim_matrix.dim() == 2:
            sim_matrix = sim_matrix.unsqueeze(0)
        row_max = sim_matrix.max(dim=2).values
        row_mean = sim_matrix.mean(dim=2)
        col_max = sim_matrix.max(dim=1).values
        col_mean = sim_matrix.mean(dim=1)
        global_max = sim_matrix.amax(dim=(1, 2)).unsqueeze(1)
        global_mean = sim_matrix.mean(dim=(1, 2)).unsqueeze(1)
        global_std = sim_matrix.std(dim=(1, 2), unbiased=False).unsqueeze(1)
        global_min = sim_matrix.amin(dim=(1, 2)).unsqueeze(1)
        global_stats = torch.cat([global_max, global_mean, global_std, global_min], dim=1)
        features = torch.cat([row_max, row_mean, col_max, col_mean, global_stats], dim=1)
        return features

    def forward(self, sim_matrix: Tensor) -> Tensor:
        features = self.extract_features(sim_matrix)
        return self.mlp(features).squeeze(-1)


class CardinalityHead(nn.Module):
    """查询级数量预测头：输入轻量 query 特征，输出 0..Kmax 的分类 logits。"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, k_max: int = 20) -> None:
        super().__init__()
        self.k_max = k_max
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_max + 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.mlp(features)


class LearnedFieldScorer:
    """可学习语义打分器：用 TinyReranker 替代 rule-based semantic_score。"""

    def __init__(self, reranker_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = TinyReranker().to(self.device)
        self.cardinality_head: Optional[CardinalityHead] = None
        try:
            state = torch.load(reranker_path, map_location=self.device, weights_only=True)
        except TypeError:
            # 兼容旧版本 PyTorch（不支持 weights_only 参数）
            state = torch.load(reranker_path, map_location=self.device)
        state_dict = state.get("model_state", state)
        self.model.load_state_dict(state_dict, strict=False)
        cardinality_state = state.get("cardinality_state") if isinstance(state, dict) else None
        if cardinality_state:
            card_meta = state.get("cardinality_meta", {})
            self.cardinality_head = CardinalityHead(
                input_dim=int(card_meta.get("input_dim", 3)),
                hidden_dim=int(card_meta.get("hidden_dim", 16)),
                k_max=int(card_meta.get("k_max", 20)),
            ).to(self.device)
            self.cardinality_head.load_state_dict(cardinality_state)
            self.cardinality_head.eval()
        self.model.eval()

    def score(self, q_vecs: List[Vector], m_vecs: List[Vector]) -> Tuple[float, Dict[str, List[float]]]:
        with torch.no_grad():
            sim_matrix = compute_sim_matrix(q_vecs, m_vecs).to(self.device)
            raw_score = self.model(sim_matrix.unsqueeze(0)).item()

        sigmoid_score = float(torch.sigmoid(torch.tensor(raw_score)).item())
        # 关键修复：推理主链路直接使用 raw score，避免 sigmoid 饱和把分差压扁。
        semantic_score = float(raw_score) / 50.0
        debug = {
            "raw_score": [raw_score],
            "sigmoid_score": [sigmoid_score],
            "semantic_score": [semantic_score],
        }
        return semantic_score, debug

    def predict_cardinality(self, query_features: List[float]) -> Optional[int]:
        if self.cardinality_head is None:
            return None
        with torch.no_grad():
            tensor = torch.tensor(query_features, dtype=torch.float32, device=self.device)
            logits = self.cardinality_head(tensor)
            return int(logits.argmax(dim=-1).item())
