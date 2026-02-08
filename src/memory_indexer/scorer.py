"""向量组相似度打分。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .utils import Vector, dot


class FieldScorer:
    """字段级相似度：每个 query 向量在 memory 里找最佳匹配。"""

    def __init__(self, top_k_per_q: int = 1):
        self.top_k_per_q = top_k_per_q

    def score(self, q_vecs: List[Vector], m_vecs: List[Vector]) -> Tuple[float, Dict[str, List[float]]]:
        if not q_vecs or not m_vecs:
            return 0.0, {"best_per_q": [], "top_used": []}

        best_per_q = []
        for q in q_vecs:
            best = max(dot(q, m) for m in m_vecs)
            best_per_q.append(best)

        k = min(self.top_k_per_q, len(best_per_q))
        top_used = sorted(best_per_q)[-k:] if k else best_per_q
        score = sum(top_used) / len(top_used) if top_used else 0.0
        debug = {"best_per_q": best_per_q, "top_used": top_used}
        return score, debug


def compute_sim_matrix(q_vecs: List[Vector], m_vecs: List[Vector]):
    """延迟导入 PyTorch 的 learned scorer 入口。"""

    from .learned_scorer import compute_sim_matrix as _compute_sim_matrix

    return _compute_sim_matrix(q_vecs, m_vecs)


class TinyReranker:
    """延迟构造真正的 TinyReranker(nn.Module)。"""

    def __new__(cls, *args, **kwargs):
        from .learned_scorer import TinyReranker as _TinyReranker

        return _TinyReranker(*args, **kwargs)


class LearnedFieldScorer:
    """延迟构造真正的 LearnedFieldScorer。"""

    def __new__(cls, *args, **kwargs):
        from .learned_scorer import LearnedFieldScorer as _LearnedFieldScorer

        return _LearnedFieldScorer(*args, **kwargs)
