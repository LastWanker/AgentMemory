"""向量组构造器。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from .utils import Vector


class Vectorizer:
    """根据策略把 token 向量转成向量组。"""

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
        # 中文注释：简单基于词频和长度打分，模拟 TF-IDF 选择
        freq = Counter(tokens)
        scored = []
        for idx, token in enumerate(tokens):
            score = len(token) + freq[token]
            scored.append((score, idx, token))
        scored.sort(reverse=True)
        selected = scored[:k] if scored else []
        idxs = [idx for _, idx, _ in selected]
        group = [token_vecs[i] for i in idxs] if idxs else token_vecs[:1]
        aux = {"selected_tokens": [tokens[i] for i in idxs]}
        return group, aux

    def _cluster_centers(
        self, token_vecs: List[Vector], tokens: List[str], r: int
    ) -> Tuple[List[Vector], Dict[str, List[str]]]:
        # 中文注释：占位式“聚类”，实际按均匀间隔抽样
        if not token_vecs:
            return [], {"center_tokens": []}
        step = max(1, len(token_vecs) // r)
        idxs = list(range(0, len(token_vecs), step))[:r]
        group = [token_vecs[i] for i in idxs]
        aux = {"center_tokens": [tokens[i] for i in idxs]}
        return group, aux
