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
        self._mem_fixed_cache: Dict[int, Tensor] = {}
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

    def _to_fixed_matrix(self, vecs: List[Vector], size: int = 8) -> Tensor:
        """将可变长向量组转为固定大小矩阵，便于批量计算。"""

        if not vecs:
            return torch.zeros((size, 1), dtype=torch.float32)
        tensor = torch.tensor(vecs[:size], dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        rows = min(size, tensor.shape[0])
        cols = max(1, tensor.shape[1] if tensor.dim() > 1 else 1)
        fixed = torch.zeros((size, cols), dtype=torch.float32)
        fixed[:rows, :cols] = tensor[:rows, :cols]
        return fixed

    def _memory_fixed_matrix(self, m_vecs: List[Vector], size: int = 8) -> Tensor:
        cache_key = id(m_vecs)
        cached = self._mem_fixed_cache.get(cache_key)
        if cached is not None:
            return cached
        fixed = self._to_fixed_matrix(m_vecs, size=size)
        if self.mem_cache_limit <= 0 or len(self._mem_fixed_cache) < self.mem_cache_limit:
            self._mem_fixed_cache[cache_key] = fixed
        return fixed

    def score(self, q_vecs: List[Vector], m_vecs: List[Vector]) -> Tuple[float, Dict[str, List[float]]]:
        with torch.no_grad():
            q_fixed = self._to_fixed_matrix(q_vecs)
            m_fixed = self._memory_fixed_matrix(m_vecs)
            q_dim = q_fixed.shape[1]
            if m_fixed.shape[1] == q_dim:
                m_ready = m_fixed
            else:
                m_ready = torch.zeros((8, q_dim), dtype=torch.float32)
                cols = min(q_dim, m_fixed.shape[1])
                m_ready[:, :cols] = m_fixed[:, :cols]
            sim_matrix = torch.matmul(q_fixed, m_ready.t()).to(self.device)
            raw_tensor = self.model(sim_matrix.unsqueeze(0))
            raw_score = float(raw_tensor.squeeze(0).item())

        sigmoid_score = float(torch.sigmoid(torch.tensor(raw_score)).item())
        # 关键修复：推理主链路直接使用 raw score，避免 sigmoid 饱和把分差压扁。
        semantic_score = float(raw_score) / 50.0
        debug = {
            "raw_score": [raw_score],
            "sigmoid_score": [sigmoid_score],
            "semantic_score": [semantic_score],
        }
        return semantic_score, debug

    def score_many(
        self,
        q_vecs: List[Vector],
        mem_vecs_list: List[List[Vector]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, List[float]]]]:
        """Batch scoring for one query against multiple memory candidates."""

        if not mem_vecs_list:
            return []
        effective_batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        outputs: List[Tuple[float, Dict[str, List[float]]]] = []
        if not q_vecs:
            empty = {
                "raw_score": [0.0],
                "sigmoid_score": [0.5],
                "semantic_score": [0.0],
            }
            return [(0.0, empty) for _ in mem_vecs_list]

        size = 8
        q_fixed = self._to_fixed_matrix(q_vecs, size=size)
        q_dim = q_fixed.shape[1]

        with torch.no_grad():
            for start in range(0, len(mem_vecs_list), effective_batch_size):
                chunk = mem_vecs_list[start : start + effective_batch_size]
                fixed_mems = [self._memory_fixed_matrix(m_vecs, size=size) for m_vecs in chunk]
                same_dim = all(mem.shape[1] == q_dim for mem in fixed_mems)
                if same_dim:
                    mem_batch = torch.stack(fixed_mems, dim=0)
                else:
                    mem_batch = torch.zeros((len(chunk), size, q_dim), dtype=torch.float32)
                    for idx, mem in enumerate(fixed_mems):
                        cols = min(q_dim, mem.shape[1])
                        mem_batch[idx, :, :cols] = mem[:, :cols]

                q_batch = q_fixed.unsqueeze(0).expand(len(chunk), -1, -1)
                if self.device.type != "cpu":
                    q_batch = q_batch.to(self.device, non_blocking=True)
                    mem_batch = mem_batch.to(self.device, non_blocking=True)
                sim_batch = torch.matmul(q_batch, mem_batch.transpose(1, 2))
                raw_tensor = self.model(sim_batch).detach().cpu()
                sigmoid_tensor = torch.sigmoid(raw_tensor)
                semantic_tensor = raw_tensor / 50.0
                raw_scores = raw_tensor.tolist()
                sigmoid_scores = sigmoid_tensor.tolist()
                semantic_scores = semantic_tensor.tolist()
                for raw, sigmoid_score, semantic_score in zip(
                    raw_scores, sigmoid_scores, semantic_scores
                ):
                    outputs.append(
                        (
                            float(semantic_score),
                            {
                                "raw_score": [float(raw)],
                                "sigmoid_score": [float(sigmoid_score)],
                                "semantic_score": [float(semantic_score)],
                            },
                        )
                    )
        return outputs

    def predict_cardinality(self, query_features: List[float]) -> Optional[int]:
        if self.cardinality_head is None:
            return None
        with torch.no_grad():
            tensor = torch.tensor(query_features, dtype=torch.float32, device=self.device)
            logits = self.cardinality_head(tensor)
            return int(logits.argmax(dim=-1).item())
