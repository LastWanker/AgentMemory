from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import httpx

from .config import ChatAppConfig
from .models import MemoryRef
from .suppressor_adapter import SuppressorAdapter


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class RetrievalBundle:
    coarse_refs: list[MemoryRef]
    association_refs: list[MemoryRef]
    association_tags: list[dict]
    association_trace: dict
    suppressor_trace: dict
    retrieval_label: str


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
        self._local_coarse_retriever = None
        self._local_association_retriever = None
        self._suppressor = SuppressorAdapter(config)

    def retrieve(self, query: str, top_k: int) -> List[MemoryRef]:
        return self.retrieve_bundle(query, top_k).coarse_refs

    def retrieve_bundle(
        self,
        query: str,
        top_k: int,
        *,
        memory_preference_enabled: bool | None = None,
    ) -> RetrievalBundle:
        suppressor_enabled = (
            bool(self._config.suppressor_enabled)
            if memory_preference_enabled is None
            else bool(memory_preference_enabled)
        )
        fetch_top_k = int(top_k) + (
            int(self._config.suppressor_extra_candidates) if suppressor_enabled else 0
        )
        fetch_top_k = max(1, fetch_top_k)
        coarse_refs = self._retrieve_coarse(query, fetch_top_k)
        association_refs: list[MemoryRef] = []
        association_trace: dict = {}
        association_tags: list[dict] = []
        try:
            association_refs, association_trace = self._retrieve_association(
                query,
                top_k=fetch_top_k,
                exclude_memory_ids={ref.memory_id for ref in coarse_refs},
            )
            association_tags = self._extract_association_tags(association_trace)
        except Exception:
            association_refs = []
            association_trace = {}
            association_tags = []
        coarse_refs, coarse_suppress_trace = self._suppressor.apply(
            query,
            "coarse",
            coarse_refs,
            enabled_override=memory_preference_enabled,
        )
        association_refs, association_suppress_trace = self._suppressor.apply(
            query,
            "association",
            association_refs,
            enabled_override=memory_preference_enabled,
        )
        suppressor_trace = {
            "enabled": suppressor_enabled,
            "override": memory_preference_enabled,
            "fetch_top_k": int(fetch_top_k),
            "output_top_k": int(top_k),
            "backend": str((coarse_suppress_trace or {}).get("backend") or (association_suppress_trace or {}).get("backend") or ""),
            "lanes": {
                "coarse": coarse_suppress_trace,
                "association": association_suppress_trace,
            },
        }
        coarse_refs = coarse_refs[: max(1, int(top_k))]
        association_refs = association_refs[: max(1, int(top_k))]
        retrieval_label = "coarse+association" if association_refs or association_tags else "coarse-only"
        if any(int((lane or {}).get("applied_count", 0)) > 0 for lane in suppressor_trace["lanes"].values()):
            retrieval_label += "+suppressor"
        return RetrievalBundle(
            coarse_refs=coarse_refs,
            association_refs=association_refs,
            association_tags=association_tags,
            association_trace=association_trace,
            suppressor_trace=suppressor_trace,
            retrieval_label=retrieval_label,
        )

    def _retrieve_coarse(self, query: str, top_k: int) -> List[MemoryRef]:
        if self._config.retrieval_mode == "http":
            try:
                hits = self._retrieve_http(query, top_k)
                if hits:
                    return hits
            except Exception:
                pass
        try:
            hits = self._retrieve_local_project(query, top_k)
            if hits:
                return hits
        except Exception:
            pass
        return self._retrieve_bundle_fallback(query, top_k)

    def _retrieve_http(self, query: str, top_k: int) -> List[MemoryRef]:
        url = f"{self._config.retrieval_url}/retrieve"
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"query": query, "top_k": top_k})
            resp.raise_for_status()
        payload = resp.json()
        hits = payload.get("hits", [])
        out: list[MemoryRef] = []
        for hit in hits:
            ref = MemoryRef.model_validate(hit)
            if float(ref.base_score) == 0.0 and float(ref.score) != 0.0:
                ref.base_score = float(ref.score)
            out.append(ref)
        return out

    def _retrieve_bundle_fallback(self, query: str, top_k: int) -> List[MemoryRef]:
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
                        base_score=float(score),
                        source="bundle_fallback_coarse",
                        display_text=display_text,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ref for _, ref in scored[:top_k]]

    def _retrieve_local_project(self, query: str, top_k: int) -> List[MemoryRef]:
        retriever = self._get_local_project_retriever()
        hits, _trace = retriever.retrieve(query, top_k)
        return [
            MemoryRef(
                memory_id=hit.memory_id,
                cluster_id=hit.cluster_id,
                score=hit.score,
                base_score=hit.score,
                source=f"local_v5:{hit.source}",
                display_text=hit.display_text,
            )
            for hit in hits
        ]

    def _retrieve_association(
        self,
        query: str,
        *,
        top_k: int,
        exclude_memory_ids: set[str],
    ) -> tuple[list[MemoryRef], dict]:
        retriever = self._get_local_association_retriever()
        debug = retriever.retrieve_debug(query, top_k=top_k, exclude_memory_ids=exclude_memory_ids)
        hits = [
            MemoryRef(
                memory_id=hit.memory_id,
                cluster_id=hit.cluster_id,
                score=hit.score,
                base_score=hit.score,
                source=f"local_v5:{hit.source}",
                display_text=hit.display_text,
            )
            for hit in debug.get("hits", [])
        ]
        trace = dict(debug.get("trace") or {})
        trace["activation_view"] = self._build_activation_view(trace)
        return hits, trace

    def _get_local_project_retriever(self):
        if self._local_coarse_retriever is not None:
            return self._local_coarse_retriever
        project_root = Path(__file__).resolve().parents[3]
        project_src = project_root / "src"
        if str(project_src) not in sys.path:
            sys.path.insert(0, str(project_src))
        from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever

        self._local_coarse_retriever = HybridRetriever.from_config(self._config.retrieval_config)
        return self._local_coarse_retriever

    def _get_local_association_retriever(self):
        if self._local_association_retriever is not None:
            return self._local_association_retriever
        project_root = Path(__file__).resolve().parents[3]
        project_src = project_root / "src"
        if str(project_src) not in sys.path:
            sys.path.insert(0, str(project_src))
        from agentmemory_v3.association import AssociationRetriever

        self._local_association_retriever = AssociationRetriever.from_config(self._config.retrieval_config)
        return self._local_association_retriever

    @staticmethod
    def _extract_association_tags(trace: dict) -> list[dict]:
        rows: list[dict] = []
        for key, level in (("top_l3", "L3"), ("top_l2", "L2"), ("top_l1", "L1")):
            for item in trace.get(key, []) or []:
                rows.append(
                    {
                        "node_id": str(item.get("node_id") or ""),
                        "name": str(item.get("name") or ""),
                        "level": level,
                        "score": float(item.get("score") or 0.0),
                        "origins": list(item.get("origins") or []),
                        "via_bridge": bool(item.get("via_bridge")),
                    }
                )
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        return rows[:12]

    @staticmethod
    def _build_activation_view(trace: dict) -> dict:
        node_map: dict[str, dict] = {}
        edge_map: dict[tuple[str, str, str], dict] = {}
        direct_seed_ids = {
            str(item.get("node_id") or "")
            for item in (trace.get("seed_resolution") or {}).get("accepted_seeds", [])
            if str(item.get("node_id") or "")
        }

        for key, level in (("top_l3", "L3"), ("top_l2", "L2"), ("top_l1", "L1")):
            for item in trace.get(key, []) or []:
                node_id = str(item.get("node_id") or "")
                if not node_id:
                    continue
                node_map[node_id] = {
                    "node_id": node_id,
                    "name": str(item.get("name") or node_id),
                    "level": level,
                    "score": float(item.get("score") or 0.0),
                    "origins": list(item.get("origins") or []),
                    "via_bridge": bool(item.get("via_bridge")),
                    "direct_seed": node_id in direct_seed_ids,
                }

        for packet in trace.get("packet_trace_sample", []) or []:
            node_id = str(packet.get("node_id") or "")
            if not node_id:
                continue
            state = node_map.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "name": str(packet.get("name") or node_id),
                    "level": str(packet.get("level") or ""),
                    "score": 0.0,
                    "origins": [],
                    "via_bridge": False,
                    "direct_seed": node_id in direct_seed_ids,
                },
            )
            state["name"] = str(packet.get("name") or state["name"] or node_id)
            state["level"] = str(packet.get("level") or state["level"] or "")
            state["score"] = max(float(state["score"]), float(packet.get("score") or 0.0))
            origin = str(packet.get("origin_type") or "")
            if origin and origin not in state["origins"]:
                state["origins"].append(origin)
            if origin == "from_bridge":
                state["via_bridge"] = True
            if node_id in direct_seed_ids:
                state["direct_seed"] = True
            came_from = str(packet.get("came_from_node_id") or "")
            if not came_from:
                continue
            kind = ""
            if origin == "from_child":
                kind = "ascend"
            elif origin == "from_parent":
                kind = "descend"
            elif origin == "from_bridge":
                kind = "bridge"
            if not kind:
                continue
            edge_key = (came_from, node_id, kind)
            edge = edge_map.setdefault(
                edge_key,
                {
                    "source_id": came_from,
                    "target_id": node_id,
                    "kind": kind,
                    "score": 0.0,
                },
            )
            edge["score"] = max(float(edge["score"]), float(packet.get("score") or 0.0))

        nodes = sorted(
            node_map.values(),
            key=lambda item: (
                {"L3": 0, "L2": 1, "L1": 2}.get(str(item.get("level") or "").upper(), 9),
                -float(item.get("score") or 0.0),
                str(item.get("name") or ""),
            ),
        )
        edges = sorted(
            edge_map.values(),
            key=lambda item: (
                {"descend": 0, "ascend": 1, "bridge": 2}.get(str(item.get("kind") or ""), 9),
                -float(item.get("score") or 0.0),
            ),
        )
        return {
            "query": str(trace.get("query") or ""),
            "activation_counts": trace.get("activation_counts") or {},
            "nodes": nodes,
            "edges": edges,
            "direct_seed_ids": sorted(direct_seed_ids),
        }

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
