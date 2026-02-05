"""向量组构造器。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import re

from .utils import Vector, dot, norm


class Vectorizer:
    """根据策略把 token 向量转成向量组。"""

    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")

    def __init__(self, strategy: str, k: int = 32, r: int = 6):
        self.strategy = strategy
        self.k = k
        self.r = r

    def make_group(
        self, token_vecs: List[Vector], tokens: List[str], text: str
    ) -> Tuple[List[Vector], Dict[str, List[str]]]:
        if self.strategy == "token_pool_topk":
            return self._token_pool_topk(token_vecs, tokens, k=self.k)
        if self.strategy == "cluster_centers_r":
            return self._cluster_centers(token_vecs, tokens, r=self.r)
        raise ValueError(f"未知策略: {self.strategy}")

    def _token_pool_topk(
        self, token_vecs: List[Vector], tokens: List[str], k: int
    ) -> Tuple[List[Vector], Dict[str, List[str]]]:
        # 基于基础过滤 + 多样性约束的 token 选择
        freq = Counter(tokens)
        scored: List[Tuple[float, int, str]] = []
        for idx, token in enumerate(tokens):
            if not self._is_valid_token(token):
                continue
            score = self._token_score(token, freq[token])
            scored.append((score, idx, token))
        if not scored:
            scored = [
                (self._token_score(token, freq[token]), idx, token)
                for idx, token in enumerate(tokens)
            ]
        scored.sort(reverse=True)
        candidate_idxs = [idx for _, idx, _ in scored]
        idxs = self._select_diverse(candidate_idxs, token_vecs, k=k)
        group = [token_vecs[i] for i in idxs] if idxs else token_vecs[:1]
        aux = {"selected_tokens": [tokens[i] for i in idxs] if idxs else []}
        return group, aux

    def _cluster_centers(
        self, token_vecs: List[Vector], tokens: List[str], r: int
    ) -> Tuple[List[Vector], Dict[str, List[str]]]:
        # 占位式“聚类”，实际按均匀间隔抽样
        if not token_vecs:
            return [], {"center_tokens": []}
        step = max(1, len(token_vecs) // r)
        idxs = list(range(0, len(token_vecs), step))[:r]
        group = [token_vecs[i] for i in idxs]
        aux = {"center_tokens": [tokens[i] for i in idxs]}
        return group, aux

    def _token_score(self, token: str, freq: int) -> float:
        length_score = float(len(token))
        repetition_penalty = 1.0 / (1.0 + float(freq))
        return length_score + repetition_penalty

    def _is_valid_token(self, token: str) -> bool:
        stripped = token.strip()
        if not stripped:
            return False
        if self._CJK_RE.search(stripped):
            return True
        if len(stripped) < 2:
            return False
        return any(ch.isalnum() for ch in stripped)

    def _select_diverse(self, candidates: List[int], token_vecs: List[Vector], k: int) -> List[int]:
        if not candidates or not token_vecs:
            return []
        norms = [norm(vec) or 1e-12 for vec in token_vecs]
        thresholds = [0.85, 0.92, 0.97, 1.0]
        selected: List[int] = []

        def max_similarity(idx: int) -> float:
            if not selected:
                return 0.0
            sims = [
                dot(token_vecs[idx], token_vecs[sel_idx]) / (norms[idx] * norms[sel_idx])
                for sel_idx in selected
            ]
            return max(sims) if sims else 0.0

        for threshold in thresholds:
            for idx in candidates:
                if idx in selected:
                    continue
                if not selected or max_similarity(idx) < threshold:
                    selected.append(idx)
                if len(selected) >= k:
                    return selected[:k]

        if len(selected) < k:
            for idx in candidates:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= k:
                    break
        return selected[:k]
