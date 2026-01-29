"""粗召回与词法索引。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import math

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


class LexicalIndex:
    """词法索引（倒排），提供“第二路证据”。"""

    def __init__(self) -> None:
        self.token_to_docs: Dict[str, List[str]] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_count: int = 0

    def add(self, mem_id: str, tokens: Iterable[str]) -> None:
        token_list = [token for token in tokens if token]
        if not token_list:
            return
        if mem_id not in self.doc_tokens:
            self.doc_count += 1
        self.doc_tokens[mem_id] = token_list
        unique_tokens = set(token_list)
        for token in unique_tokens:
            self.token_to_docs.setdefault(token, []).append(mem_id)
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(self, query_tokens: Iterable[str], top_n: int = 1000) -> List[Tuple[str, float]]:
        """返回词法候选及分数（简化版 IDF 加权 overlap）。"""

        tokens = [token for token in query_tokens if token]
        if not tokens:
            return []
        query_set = set(tokens)
        total_idf = sum(self._idf(token) for token in query_set) or 1.0

        candidate_scores: Dict[str, float] = {}
        for token in query_set:
            for mem_id in self.token_to_docs.get(token, []):
                candidate_scores.setdefault(mem_id, 0.0)
                candidate_scores[mem_id] += self._idf(token)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[: min(top_n, len(ranked))]
        return [(mem_id, float(score / total_idf)) for mem_id, score in top]

    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        return math.log((1 + self.doc_count) / (1 + df)) + 1.0
