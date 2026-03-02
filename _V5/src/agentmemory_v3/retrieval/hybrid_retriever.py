from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.retrieval.bm25_index import BM25Index
from agentmemory_v3.retrieval.dense_index import DenseIndex
from agentmemory_v3.retrieval.interfaces import RetrieveResult
from agentmemory_v3.utils.io import read_jsonl


@dataclass
class RetrievalArtifacts:
    dense_index: DenseIndex
    bm25_index: BM25Index
    memory_rows: list[dict]
    mem_id_to_idx: dict[str, int]
    clusters: dict[str, list[str]]
    mem_id_to_cluster: dict[str, str]
    dense_top_n: int
    bm25_top_n: int
    expanded_top_n_cap: int
    cluster_expand_ratio: float
    cluster_expand_min_hits: int


class HybridRetriever:
    def __init__(self, artifacts: RetrievalArtifacts) -> None:
        self.artifacts = artifacts

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HybridRetriever":
        cfg = load_yaml_config(config_path)
        root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
        dense_index = DenseIndex.load(root_dir / "indexes" / "dense_artifact.pkl", root_dir / "indexes" / "dense_matrix.npy")
        bm25_index = BM25Index.load(root_dir / "indexes" / "bm25.pkl")
        memory_rows = list(read_jsonl(root_dir / "processed" / "memory.jsonl"))
        mem_id_to_idx = {row["memory_id"]: idx for idx, row in enumerate(memory_rows)}
        clusters: dict[str, list[str]] = defaultdict(list)
        mem_id_to_cluster: dict[str, str] = {}
        for row in memory_rows:
            cluster_id = str(row.get("cluster_id") or row["memory_id"])
            clusters[cluster_id].append(row["memory_id"])
            mem_id_to_cluster[row["memory_id"]] = cluster_id
        return cls(
            RetrievalArtifacts(
                dense_index=dense_index,
                bm25_index=bm25_index,
                memory_rows=memory_rows,
                mem_id_to_idx=mem_id_to_idx,
                clusters=dict(clusters),
                mem_id_to_cluster=mem_id_to_cluster,
                dense_top_n=int(cfg_get(cfg, "retrieval.dense_top_n", 50)),
                bm25_top_n=int(cfg_get(cfg, "retrieval.bm25_top_n", 50)),
                expanded_top_n_cap=int(cfg_get(cfg, "retrieval.expanded_top_n_cap", 150)),
                cluster_expand_ratio=float(cfg_get(cfg, "retrieval.cluster_expand_ratio", 0.5)),
                cluster_expand_min_hits=int(cfg_get(cfg, "retrieval.cluster_expand_min_hits", 2)),
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_vec: np.ndarray | None = None,
    ) -> tuple[list[RetrieveResult], dict]:
        debug = self.retrieve_debug(
            query,
            top_k=top_k,
            query_vec=query_vec,
        )
        return debug["all_hits"][: max(1, int(top_k))], debug["trace"]

    def retrieve_debug(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_vec: np.ndarray | None = None,
    ) -> dict:
        candidate_pack = self.collect_candidate_pack_with_vector(query, query_vec=query_vec)
        hits = self._coarse_results(
            candidate_pack["expanded_ids"],
            candidate_pack["source_map"],
            candidate_pack["coarse_scores"],
        )
        trace = dict(candidate_pack["trace"])
        trace["retrieval_mode"] = "coarse_only"
        return {
            "hits": hits[: max(1, int(top_k))],
            "all_hits": hits,
            "trace": trace,
            "base_ids": candidate_pack["base_ids"],
            "expanded_ids": candidate_pack["expanded_ids"],
            "source_map": candidate_pack["source_map"],
            "coarse_scores": candidate_pack["coarse_scores"],
        }

    def collect_candidate_pack(self, query: str) -> dict:
        return self.collect_candidate_pack_with_vector(query, query_vec=None)

    def collect_candidate_pack_with_vector(self, query: str, query_vec: np.ndarray | None = None) -> dict:
        if query_vec is not None:
            dense_hits = self.artifacts.dense_index.search_vector(query_vec, self.artifacts.dense_top_n)
        else:
            dense_hits = self.artifacts.dense_index.search(query, self.artifacts.dense_top_n)
        bm25_hits = self.artifacts.bm25_index.search(query, self.artifacts.bm25_top_n)
        base_ids, source_map, coarse_scores = self._merge_candidates(dense_hits, bm25_hits)
        expanded_ids = self._expand_by_cluster(base_ids)
        trace = {
            "dense_hits": len(dense_hits),
            "bm25_hits": len(bm25_hits),
            "base_candidates": len(base_ids),
            "expanded_candidates": len(expanded_ids),
        }
        return {
            "base_ids": base_ids,
            "expanded_ids": expanded_ids,
            "source_map": source_map,
            "coarse_scores": coarse_scores,
            "trace": trace,
        }

    def _coarse_results(
        self,
        candidate_ids: list[str],
        source_map: dict[str, str],
        coarse_scores: dict[str, dict[str, float]],
    ) -> list[RetrieveResult]:
        results: list[RetrieveResult] = []
        for rank, mem_id in enumerate(candidate_ids, start=1):
            idx = self.artifacts.mem_id_to_idx[mem_id]
            row = self.artifacts.memory_rows[idx]
            score_info = coarse_scores.get(mem_id, {})
            source = source_map.get(mem_id, "")
            if not source:
                source = "cluster_expand"
            results.append(
                RetrieveResult(
                    memory_id=mem_id,
                    cluster_id=str(row.get("cluster_id") or mem_id),
                    score=float(self._coarse_display_score(rank, score_info)),
                    source=f"coarse:{source}",
                    display_text=str(row.get("raw_text") or row.get("text") or ""),
                )
            )
        return results

    @staticmethod
    def _coarse_display_score(rank: int, score_info: dict[str, float]) -> float:
        dense_rank = float(score_info.get("dense_rank", 9999.0))
        bm25_rank = float(score_info.get("bm25_rank", 9999.0))
        dense_rrf = 0.0 if dense_rank >= 9999.0 else 1.0 / (60.0 + dense_rank)
        bm25_rrf = 0.0 if bm25_rank >= 9999.0 else 1.0 / (60.0 + bm25_rank)
        dense_score = float(score_info.get("dense", 0.0))
        bm25_score = float(score_info.get("bm25", 0.0))
        return dense_rrf + bm25_rrf + 0.01 * dense_score + 0.01 * bm25_score + 1e-6 / max(1, rank)

    def _merge_candidates(
        self,
        dense_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[int, float]],
    ) -> tuple[list[str], dict[str, str], dict[str, dict[str, float]]]:
        source_map: dict[str, str] = {}
        coarse_scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {"dense": 0.0, "bm25": 0.0, "dense_rank": 9999.0, "bm25_rank": 9999.0}
        )
        order: list[str] = []
        for rank, (idx, score) in enumerate(dense_hits, start=1):
            mem_id = self.artifacts.memory_rows[idx]["memory_id"]
            if mem_id not in source_map:
                order.append(mem_id)
                source_map[mem_id] = "dense"
            coarse_scores[mem_id]["dense"] = float(score)
            coarse_scores[mem_id]["dense_rank"] = float(rank)
        for rank, (idx, score) in enumerate(bm25_hits, start=1):
            mem_id = self.artifacts.memory_rows[idx]["memory_id"]
            if mem_id not in source_map:
                order.append(mem_id)
                source_map[mem_id] = "bm25"
            elif source_map[mem_id] == "dense":
                source_map[mem_id] = "both"
            coarse_scores[mem_id]["bm25"] = float(score)
            coarse_scores[mem_id]["bm25_rank"] = float(rank)
        return order, source_map, coarse_scores

    def _expand_by_cluster(self, base_ids: list[str]) -> list[str]:
        cluster_hits: Counter[str] = Counter()
        for mem_id in base_ids:
            cluster_hits[self.artifacts.mem_id_to_cluster.get(mem_id, mem_id)] += 1
        expanded = list(base_ids)
        seen = set(expanded)
        for cluster_id, hit_count in cluster_hits.items():
            members = self.artifacts.clusters.get(cluster_id, [])
            if not members:
                continue
            if hit_count < self.artifacts.cluster_expand_min_hits:
                continue
            if hit_count / len(members) < self.artifacts.cluster_expand_ratio:
                continue
            for mem_id in members:
                if mem_id in seen:
                    continue
                expanded.append(mem_id)
                seen.add(mem_id)
                if len(expanded) >= self.artifacts.expanded_top_n_cap:
                    return expanded[: self.artifacts.expanded_top_n_cap]
        return expanded[: self.artifacts.expanded_top_n_cap]
