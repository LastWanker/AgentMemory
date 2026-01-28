"""粗召回索引。"""

from __future__ import annotations

from typing import List, Tuple

from .utils import Vector, dot


class CoarseIndex:
    """粗召回索引：MVP 用暴力搜索即可。"""

    def __init__(self) -> None:
        self.mem_ids: List[str] = []
        self.coarse_vecs: List[Vector] = []

    def add(self, mem_id: str, coarse_vec: Vector) -> None:
        self.mem_ids.append(mem_id)
        self.coarse_vecs.append(coarse_vec)

    def search(self, q_coarse: Vector, top_n: int = 1000) -> List[Tuple[str, float]]:
        # 向量已归一化，点积就是余弦相似度
        scores = [dot(vec, q_coarse) for vec in self.coarse_vecs]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[: min(top_n, len(ranked))]
        return [(self.mem_ids[i], float(score)) for i, score in top]
