"""检索器：粗召回 + 精排。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

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
        # 用于“同一 query 重复跑”的一致性监测：记录上一次的候选签名与 top-k。
        self._last_signature: Optional[str] = None
        self._last_top_ids: List[str] = []

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

        metrics, explain = self._build_metrics_and_explain(ranked, weights)
        return RouteOutput(
            policy=self.policy,
            weights=weights,
            selected_ids=selected,
            metrics=metrics,
            explain=explain,
        )

    def _softmax_weights(self, ranked: List[Tuple[str, float]]) -> Dict[str, float]:
        scores = [score for _, score in ranked]
        if not scores:
            return {}
        max_score = max(scores)
        exps = [pow(2.718281828, score - max_score) for score in scores]
        total = sum(exps) if exps else 1.0
        return {mem_id: exp / total for (mem_id, _), exp in zip(ranked, exps)}

    def _build_metrics_and_explain(
        self,
        ranked: List[Tuple[str, float]],
        weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """计算路由指标，提供可解释信息。

        指标说明：
        - entropy：权重分布熵（越低越尖锐）。
        - mass_at_k：前 k 个权重占比（越高越集中）。
        - consistency：同一候选签名的 top-k 重复一致性。
        - counterfactual_drop_top1/topk：移除最高权重候选后 top 分数下降幅度。
        """

        if not ranked:
            return {
                "entropy": 0.0,
                "mass_at_k": 0.0,
                "consistency": 1.0,
                "counterfactual_drop_top1": 0.0,
                "counterfactual_drop_topk": 0.0,
            }, {"note": ["无候选，指标返回默认值"]}

        ordered_mem_ids = [mem_id for mem_id, _ in ranked]
        ordered_scores = [score for _, score in ranked]
        ordered_weights = [weights.get(mem_id, 0.0) for mem_id in ordered_mem_ids]

        # 归一化权重，避免除零；硬/半硬路由时也能得到“分布感”。
        weight_total = sum(ordered_weights) or 1e-12
        probs = [w / weight_total for w in ordered_weights]

        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        k = min(self.top_k, len(probs))
        mass_at_k = sum(probs[:k]) if k > 0 else 0.0

        signature = self._signature_for_ranked(ranked)
        if signature == self._last_signature and self._last_top_ids:
            overlap = len(set(self._last_top_ids[:k]) & set(ordered_mem_ids[:k]))
            consistency = overlap / (k or 1)
            consistency_note = "同一候选签名重复运行，计算 top-k 重叠率"
        else:
            consistency = 1.0
            consistency_note = "首次或候选签名变化，默认一致性为 1.0"

        self._last_signature = signature
        self._last_top_ids = ordered_mem_ids[:k]

        counterfactual_drop_top1, counterfactual_drop_topk = self._counterfactual_drop(
            ordered_scores,
            k,
        )

        metrics = {
            "entropy": float(entropy),
            "mass_at_k": float(mass_at_k),
            "consistency": float(consistency),
            "counterfactual_drop_top1": float(counterfactual_drop_top1),
            "counterfactual_drop_topk": float(counterfactual_drop_topk),
        }
        explain = {
            "top_ranked_ids": ordered_mem_ids[:k],
            "consistency_note": [consistency_note],
        }
        return metrics, explain

    def _signature_for_ranked(self, ranked: List[Tuple[str, float]]) -> str:
        """构建候选签名，用于同一 query 重复运行的一致性判断。"""

        # 采用分桶后的签名：对分数做轻微量化，减少接近分数的排序抖动。
        # 同桶内只保留 mem_id 集合（排序后拼接），避免因顺序交换导致签名变化。
        buckets: Dict[float, List[str]] = {}
        for mem_id, score in ranked:
            bucket_key = round(score, 3)
            buckets.setdefault(bucket_key, []).append(mem_id)

        signature_parts: List[str] = []
        for bucket_key in sorted(buckets.keys(), reverse=True):
            mem_ids = ",".join(sorted(buckets[bucket_key]))
            signature_parts.append(f"{bucket_key:.3f}:{mem_ids}")
        return "|".join(signature_parts)

    def _counterfactual_drop(self, ordered_scores: List[float], k: int) -> Tuple[float, float]:
        """计算反事实敏感性：移除最高分后 top-1 / top-k 分数下降幅度。"""

        if not ordered_scores:
            return 0.0, 0.0

        original_top1 = ordered_scores[0]
        original_topk = sum(ordered_scores[:k]) / (k or 1)
        # 移除最高分，模拟“最强记忆不可见”
        dropped_scores = ordered_scores[1:]
        if not dropped_scores:
            return original_top1, original_topk

        new_top1 = dropped_scores[0]
        new_topk = sum(dropped_scores[:k]) / (k or 1)
        return original_top1 - new_top1, original_topk - new_topk


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
