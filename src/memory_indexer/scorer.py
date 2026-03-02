"""向量组相似度打分。"""

from __future__ import annotations

from pathlib import Path
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


class BipartiteAlignTransformer:
    """延迟构造 bipartite align transformer。"""

    def __new__(cls, *args, **kwargs):
        from .learned_scorer_bipartite import BipartiteAlignTransformer as _BipartiteAlignTransformer

        return _BipartiteAlignTransformer(*args, **kwargs)


class BipartiteLearnedFieldScorer:
    """延迟构造 bipartite learned scorer。"""

    def __new__(cls, *args, **kwargs):
        from .learned_scorer_bipartite import BipartiteLearnedFieldScorer as _BipartiteLearnedFieldScorer

        return _BipartiteLearnedFieldScorer(*args, **kwargs)


def _extract_model_family(reranker_path: str) -> str:
    path = Path(reranker_path)
    if not path.exists():
        return "tiny"
    try:
        import torch  # type: ignore

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            meta = state.get("meta", {})
            if isinstance(meta, dict):
                family = str(meta.get("model_family", "")).strip().lower()
                if family:
                    return family
                if "bipartite_tau" in meta or "bipartite_input_dim" in meta:
                    return "bipartite"
    except Exception:
        return "tiny"
    return "tiny"


class LearnedFieldScorer:
    """延迟构造真正的 LearnedFieldScorer。"""

    def __new__(cls, *args, **kwargs):
        reranker_path = kwargs.get("reranker_path")
        if reranker_path is None and args:
            reranker_path = args[0]
        family = _extract_model_family(str(reranker_path)) if reranker_path else "tiny"
        if family in {"bipartite", "bipartite_align_transformer"}:
            from .learned_scorer_bipartite import (
                BipartiteLearnedFieldScorer as _LearnedFieldScorer,
            )
        else:
            from .learned_scorer import LearnedFieldScorer as _LearnedFieldScorer

        return _LearnedFieldScorer(*args, **kwargs)
