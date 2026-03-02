from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

import httpx

from .config import ChatAppConfig
from .models import MemoryRef


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _simple_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    denom = max(1, len(q_tokens))
    bonus = 1.0 if query and query in text else 0.0
    return overlap / denom + bonus


class RetrieverAdapter:
    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._local_retriever = None

    def retrieve(self, query: str, top_k: int, coarse_only: bool = False) -> List[MemoryRef]:
        if self._config.retrieval_mode == "http":
            try:
                hits = self._retrieve_http(query, top_k, coarse_only)
                if hits:
                    return hits
            except Exception:
                pass
        try:
            hits = self._retrieve_local_project(query, top_k, coarse_only)
            if hits:
                return hits
        except Exception:
            pass
        return self._retrieve_bundle(query, top_k, coarse_only)

    def _retrieve_http(self, query: str, top_k: int, coarse_only: bool) -> List[MemoryRef]:
        url = f"{self._config.retrieval_url}/retrieve"
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"query": query, "top_k": top_k, "coarse_only": coarse_only})
            resp.raise_for_status()
        payload = resp.json()
        hits = payload.get("hits", [])
        return [MemoryRef.model_validate(hit) for hit in hits]

    def _retrieve_bundle(self, query: str, top_k: int, coarse_only: bool) -> List[MemoryRef]:
        bundle_path = self._config.retrieval_bundle
        if not bundle_path.exists():
            return []
        scored: list[tuple[float, MemoryRef]] = []
        for row in self._iter_jsonl(bundle_path):
            display_text = str(row.get("display_text") or row.get("raw_text") or "")
            search_text = str(row.get("search_text") or display_text)
            score = _simple_overlap_score(query, search_text)
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    MemoryRef(
                        memory_id=str(row.get("memory_id") or ""),
                        cluster_id=str(row.get("cluster_id") or ""),
                        score=float(score),
                        source="bundle_fallback_coarse" if coarse_only else "bundle_fallback",
                        display_text=display_text,
                        slot_texts=dict(row.get("slot_texts") or {}),
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ref for _, ref in scored[:top_k]]

    def _retrieve_local_project(self, query: str, top_k: int, coarse_only: bool) -> List[MemoryRef]:
        retriever = self._get_local_project_retriever()
        hits, _trace = retriever.retrieve(query, top_k, coarse_only=coarse_only)
        return [
            MemoryRef(
                memory_id=hit.memory_id,
                cluster_id=hit.cluster_id,
                score=hit.score,
                source=f"local_{self._config.retrieval_project.lower()}:{hit.source}",
                display_text=hit.display_text,
                slot_texts=hit.slot_texts,
            )
            for hit in hits
        ]

    def _get_local_project_retriever(self):
        if self._local_retriever is not None:
            return self._local_retriever
        project_root = Path(__file__).resolve().parents[3] / ("_V4" if self._config.retrieval_project == "V4" else "_V3")
        project_src = project_root / "src"
        if str(project_src) not in sys.path:
            sys.path.insert(0, str(project_src))
        from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever

        self._local_retriever = HybridRetriever.from_config(self._config.retrieval_config)
        return self._local_retriever

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
