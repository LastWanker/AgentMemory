"""检索器：粗召回 + 精排。"""

from __future__ import annotations

from typing import List

from .index import CoarseIndex
from .models import Query, RetrieveResult
from .scorer import FieldScorer
from .store import MemoryStore


class Retriever:
    """组合粗召回索引与向量组精排。"""

    def __init__(self, store: MemoryStore, index: CoarseIndex, scorer: FieldScorer) -> None:
        self.store = store
        self.index = index
        self.scorer = scorer

    def retrieve(self, q: Query, top_n: int = 1000, top_k: int = 10) -> List[RetrieveResult]:
        if q.coarse_vec is None or q.q_vecs is None:
            raise ValueError("查询向量尚未构建")

        candidates = self.index.search(q.coarse_vec, top_n=top_n)
        results: List[RetrieveResult] = []
        for mem_id, coarse_score in candidates:
            emb = self.store.embs[mem_id]
            score, debug = self.scorer.score(q.q_vecs, emb.vecs)
            results.append(
                RetrieveResult(
                    mem_id=mem_id,
                    score=score,
                    coarse_score=coarse_score,
                    debug=debug,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
