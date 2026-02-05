"""检索器：粗召回 + 精排 + 多证据路由。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import time

from .index import CoarseIndex, LexicalIndex
from .models import Query, RetrieveResult, RouteOutput
from .scorer import FieldScorer
from .store import MemoryStore


class Router:
    """候选路由器（软/半硬/硬阶段）。"""

    def __init__(
        self,
        policy: str = "soft",
        top_k: int = 10,
        weak_weight: float = 0.05,
        fixed_channel_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.policy = policy
        self.top_k = top_k
        self.weak_weight = weak_weight
        self.fixed_channel_weights = fixed_channel_weights
        # 用于“同一 query 重复跑”的一致性监测：记录上一次的候选签名与 top-k。
        self._last_signature: Optional[str] = None
        self._last_top_ids: List[str] = []

    def route(self, scored: List[Tuple[str, Dict[str, float]]]) -> RouteOutput:
        """根据策略生成路由输出。"""
        combined_scores, channel_weights = self._combine_scores(scored)
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
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
        explain["channel_weights"] = [
            f"{key}={value:.3f}" for key, value in channel_weights.items()
        ]
        explain["feature_keys"] = [
            "semantic_score",
            "coarse_score",
            "lexical_score",
            "token_overlap",
            "source_score",
            "tag_match",
            "recency_score",
            "meta_score",
        ]
        return RouteOutput(
            policy=self.policy,
            weights=weights,
            selected_ids=selected,
            metrics=metrics,
            explain=explain,
            scores=combined_scores,
        )

    def _combine_scores(
        self, scored: List[Tuple[str, Dict[str, float]]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """根据多证据特征融合总分，并返回通道权重。"""

        channel_weights = self._compute_channel_weights(scored)
        combined: Dict[str, float] = {}
        for mem_id, features in scored:
            combined_score = (
                channel_weights["semantic"] * features.get("semantic_score", 0.0)
                + channel_weights["lexical"] * features.get("lexical_score", 0.0)
                + channel_weights["meta"] * features.get("meta_score", 0.0)
                + channel_weights["coarse"] * features.get("coarse_score", 0.0)
            )
            combined[mem_id] = float(combined_score)
        return combined, channel_weights

    def _compute_channel_weights(
        self, scored: List[Tuple[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """按 query 级别估计证据强度，得到 α/β/γ/δ。"""

        if self.fixed_channel_weights is not None:
            return self._normalize_channel_weights(self.fixed_channel_weights)

        if not scored:
            return {"semantic": 0.6, "lexical": 0.2, "meta": 0.15, "coarse": 0.05}

        def normalize_values(values: List[float]) -> List[float]:
            if not values:
                return []
            min_value = min(values)
            max_value = max(values)
            if max_value - min_value <= 1e-12:
                return [1.0 for _ in values]
            return [(value - min_value) / (max_value - min_value) for value in values]

        def avg_feature(key: str) -> float:
            raw_values = [max(0.0, features.get(key, 0.0)) for _, features in scored]
            normalized = normalize_values(raw_values)
            return sum(normalized) / len(normalized) if normalized else 0.0

        strengths = {
            "semantic": avg_feature("semantic_score"),
            "lexical": avg_feature("lexical_score"),
            "meta": avg_feature("meta_score"),
            "coarse": avg_feature("coarse_score"),
        }
        total = sum(strengths.values())
        if total <= 1e-12:
            return {"semantic": 0.6, "lexical": 0.2, "meta": 0.15, "coarse": 0.05}

        return {key: value / total for key, value in strengths.items()}

    def _normalize_channel_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """归一化通道权重，缺失键视为 0。"""

        keys = ("semantic", "lexical", "meta", "coarse")
        weights = {key: float(raw_weights.get(key, 0.0)) for key in keys}
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 1e-12:
            return {"semantic": 0.6, "lexical": 0.2, "meta": 0.15, "coarse": 0.05}
        return {key: max(0.0, value) / total for key, value in weights.items()}

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
        lexical_index: Optional[LexicalIndex] = None,
        recency_half_life_days: float = 7.0,
    ) -> None:
        self.store = store
        self.index = index
        self.scorer = scorer
        self.router = router or Router()
        self.lexical_index = lexical_index
        self.recency_half_life_days = recency_half_life_days

    def retrieve(self, q: Query, top_n: int = 1000, top_k: int = 10) -> List[RetrieveResult]:
        if q.coarse_vec is None or q.q_vecs is None:
            raise ValueError("查询向量尚未构建")

        candidates = self.index.search(q.coarse_vec, top_n=top_n)
        query_tokens = q.aux.get("lex_tokens") if q.aux else None
        if not query_tokens:
            query_tokens = q.aux.get("tokens", []) if q.aux else []
        lexical_scores = self._build_lexical_scores(query_tokens, top_n=top_n)
        scored: List[Tuple[str, Dict[str, float], float, float, Dict[str, List[float]]]] = []
        for mem_id, coarse_score in candidates:
            emb = self.store.embs[mem_id]
            score, debug = self.scorer.score(q.q_vecs, emb.vecs)
            item = self.store.items[mem_id]
            doc_tokens = self.store.tokens.get(mem_id, [])
            token_overlap = self._token_overlap(query_tokens, doc_tokens)
            lexical_score = lexical_scores.get(mem_id, token_overlap)
            source_score = self._source_score(item.source)
            tag_match = self._tag_match_score(query_tokens, item.keywords)
            recency_score = self._recency_score(item.created_at)
            meta_score = (source_score + tag_match + recency_score) / 3.0
            features = {
                "semantic_score": float(score),
                "coarse_score": float(coarse_score),
                "lexical_score": float(lexical_score),
                "token_overlap": float(token_overlap),
                "source_score": float(source_score),
                "tag_match": float(tag_match),
                "recency_score": float(recency_score),
                "meta_score": float(meta_score),
            }
            scored.append((mem_id, features, score, coarse_score, debug))

        route_output = self.router.route([(mem_id, features) for mem_id, features, _, _, _ in scored])
        results: List[RetrieveResult] = []
        for mem_id, features, score, coarse_score, debug in scored:
            if self.router.policy == "hard" and mem_id not in route_output.selected_ids:
                continue
            weight = route_output.weights.get(mem_id, 1.0)
            combined_score = route_output.scores.get(mem_id, score)
            if self.router.policy == "soft":
                final_score = combined_score
            else:
                final_score = combined_score * weight
            results.append(
                RetrieveResult(
                    mem_id=mem_id,
                    score=final_score,
                    coarse_score=coarse_score,
                    debug=debug,
                    features=features,
                    route_output=route_output,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _build_lexical_scores(self, query_tokens: List[str], top_n: int) -> Dict[str, float]:
        if not self.lexical_index:
            return {}
        return {
            mem_id: score
            for mem_id, score in self.lexical_index.search(query_tokens, top_n=top_n)
        }

    def _token_overlap(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """简单 overlap 比例，用于“词法证据”。"""

        if not query_tokens or not doc_tokens:
            return 0.0
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return len(query_set & doc_set) / (len(query_set) or 1)

    def _source_score(self, source: str) -> float:
        """对来源做轻量可信度评分。"""

        source_key = source.lower()
        if source_key in {"user", "manual"}:
            return 1.0
        if source_key in {"system", "pipeline"}:
            return 0.7
        return 0.5

    def _tag_match_score(self, query_tokens: List[str], keywords: List[str]) -> float:
        """关键词匹配：query token 命中 tags 的比例。"""

        if not query_tokens or not keywords:
            return 0.0
        query_set = set(query_tokens)
        keyword_set = set(keywords)
        return len(query_set & keyword_set) / (len(keyword_set) or 1)

    def _recency_score(self, created_at: float) -> float:
        """时间衰减，越新越接近 1.0。"""

        if created_at <= 0:
            return 0.0
        half_life_seconds = self.recency_half_life_days * 24 * 3600
        delta = max(0.0, time.time() - created_at)
        return float(pow(0.5, delta / (half_life_seconds or 1.0)))
