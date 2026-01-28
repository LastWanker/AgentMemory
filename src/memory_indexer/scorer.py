"""向量组相似度打分。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .utils import Vector, dot


class FieldScorer:
    """字段级相似度：每个 query 向量在 memory 里找最佳匹配。"""

    def __init__(self, top_k_per_q: int = 3):
        self.top_k_per_q = top_k_per_q

    def score(self, q_vecs: List[Vector], m_vecs: List[Vector]) -> Tuple[float, Dict[str, List[float]]]:
        if not q_vecs or not m_vecs:
            return 0.0, {"best_per_q": [], "top_used": []}

        best_per_q = []
        for q in q_vecs:
            # 选出与当前 q 最相似的 m
            best = max(dot(q, m) for m in m_vecs)
            best_per_q.append(best)

        k = min(self.top_k_per_q, len(best_per_q))
        top_used = sorted(best_per_q)[-k:] if k else best_per_q
        score = sum(top_used) / len(top_used) if top_used else 0.0
        debug = {"best_per_q": best_per_q, "top_used": top_used}
        return score, debug
