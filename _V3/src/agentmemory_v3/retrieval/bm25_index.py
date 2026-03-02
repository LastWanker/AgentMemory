from __future__ import annotations

import heapq
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from agentmemory_v3.utils.text import tokenize


@dataclass
class BM25Artifact:
    mem_ids: list[str]
    doc_lengths: list[int]
    avgdl: float
    token_to_postings: dict[str, list[tuple[int, int]]]
    doc_freq: dict[str, int]
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    def __init__(self, artifact: BM25Artifact) -> None:
        self.artifact = artifact

    @classmethod
    def build(cls, mem_ids: list[str], texts: list[str]) -> "BM25Index":
        token_to_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: dict[str, int] = {}
        doc_lengths: list[int] = []
        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            doc_lengths.append(len(tokens))
            freqs = Counter(tokens)
            for token, tf in freqs.items():
                token_to_postings[token].append((doc_idx, int(tf)))
                doc_freq[token] = doc_freq.get(token, 0) + 1
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        return cls(
            BM25Artifact(
                mem_ids=list(mem_ids),
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                token_to_postings=dict(token_to_postings),
                doc_freq=doc_freq,
            )
        )

    def search(self, query_text: str, top_n: int) -> list[tuple[int, float]]:
        query_tokens = list(dict.fromkeys(tokenize(query_text)))
        if not query_tokens:
            return []
        scores: dict[int, float] = defaultdict(float)
        n_docs = max(1, len(self.artifact.mem_ids))
        for token in query_tokens:
            postings = self.artifact.token_to_postings.get(token)
            if not postings:
                continue
            df = self.artifact.doc_freq.get(token, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                dl = self.artifact.doc_lengths[doc_idx]
                denom = tf + self.artifact.k1 * (1.0 - self.artifact.b + self.artifact.b * dl / max(self.artifact.avgdl, 1e-6))
                scores[doc_idx] += idf * (tf * (self.artifact.k1 + 1.0)) / max(denom, 1e-6)
        return heapq.nlargest(top_n, scores.items(), key=lambda pair: pair[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.artifact, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            artifact = pickle.load(f)
        return cls(artifact)
