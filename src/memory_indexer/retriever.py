"""检索器：粗召回 + 精排。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .index import CoarseIndex
from .models import Query, RetrieveResult, RouteOutput
from .scorer import FieldScorer
from .store import MemoryStore


class Router:
    """候选路由器（软/半硬/硬阶段）。"""

    def __init__(self, policy: str = "soft", top_k: int = 10, weak_weight: float = 0.05) -> None:
        self.policy = policy
        self.top_k = top_k
        self.weak_weight = weak_weight

    def route(self, scored: List[Tuple[str, float]]) -> RouteOutput:
        """根据策略生成路由输出。"""
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        weights: Dict[str, float] = {}
        selected: List[str] = []

        # 【新增】软路由阶段不裁剪可见集合，仅输出权重用于后续融合或解释。
        if self.policy == "soft":
            weights = self._softmax_weights(ranked)
            selected = [mem_id for mem_id, _ in ranked]
        elif self.policy == "half_hard":
            for idx, (mem_id, _) in enumerate(ranked):
                is_selected = idx < self.top_k
                if is_selected:
                    selected.append(mem_id)
                    weights[mem_id] = 1.0
                else:
                    weights[mem_id] = self.weak_weight
        elif self.policy == "hard":
            selected = [mem_id for mem_id, _ in ranked[: self.top_k]]
            for mem_id in selected:
                weights[mem_id] = 1.0
        else:
            raise ValueError(f"未知路由策略: {self.policy}")

        return RouteOutput(
            policy=self.policy,
            weights=weights,
            selected_ids=selected,
            metrics={},
            explain={},
        )

    def _softmax_weights(self, ranked: List[Tuple[str, float]]) -> Dict[str, float]:
        scores = [score for _, score in ranked]
        if not scores:
            return {}
        max_score = max(scores)
        exps = [pow(2.718281828, score - max_score) for score in scores]
        total = sum(exps) if exps else 1.0
        return {mem_id: exp / total for (mem_id, _), exp in zip(ranked, exps)}


class Retriever:
    """组合粗召回索引与向量组精排。"""

    def __init__(
        self,
        store: MemoryStore,
        index: CoarseIndex,
        scorer: FieldScorer,
        router: Optional[Router] = None,
    ) -> None:
        self.store = store
        self.index = index
        self.scorer = scorer
        self.router = router or Router()

    def retrieve(self, q: Query, top_n: int = 1000, top_k: int = 10) -> List[RetrieveResult]:
        if q.coarse_vec is None or q.q_vecs is None:
            raise ValueError("查询向量尚未构建")

        candidates = self.index.search(q.coarse_vec, top_n=top_n)
        scored: List[Tuple[str, float, float, Dict[str, List[float]]]] = []
        for mem_id, coarse_score in candidates:
            emb = self.store.embs[mem_id]
            score, debug = self.scorer.score(q.q_vecs, emb.vecs)
            scored.append((mem_id, score, coarse_score, debug))

        route_output = self.router.route([(mem_id, score) for mem_id, score, _, _ in scored])
        results: List[RetrieveResult] = []
        for mem_id, score, coarse_score, debug in scored:
            if self.router.policy == "hard" and mem_id not in route_output.selected_ids:
                continue
            weight = route_output.weights.get(mem_id, 1.0)
            results.append(
                RetrieveResult(
                    mem_id=mem_id,
                    score=score * weight,
                    coarse_score=coarse_score,
                    debug=debug,
                    route_output=route_output,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
