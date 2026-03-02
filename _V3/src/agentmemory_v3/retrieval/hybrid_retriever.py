from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.models.reranker import load_reranker, score_feature_matrix
from agentmemory_v3.models.slot_bipartite import load_slot_bipartite, score_many_slot_bipartite
from agentmemory_v3.retrieval.bm25_index import BM25Index
from agentmemory_v3.retrieval.dense_index import DenseIndex
from agentmemory_v3.retrieval.feature_builder import (
    FEATURE_NAMES,
    SLOT_FIELDS,
    build_candidate_feature_map,
    build_query_context,
    feature_vector_from_map,
    heuristic_score,
    slot_text_payload,
)
from agentmemory_v3.retrieval.interfaces import RetrieveResult
from agentmemory_v3.utils.io import read_jsonl


@dataclass
class RetrievalArtifacts:
    dense_index: DenseIndex
    bm25_index: BM25Index
    memory_rows: list[dict]
    slot_rows: dict[str, dict]
    slot_vectors: dict[str, np.ndarray]
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
        self.feature_reranker_bundle = None
        self.slot_bipartite_bundle = None
        self.reranker_family = "feature_mlp"
        self.blend_alpha = 0.35

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HybridRetriever":
        cfg = load_yaml_config(config_path)
        root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V3"))
        dense_index = DenseIndex.load(root_dir / "indexes" / "dense_artifact.pkl", root_dir / "indexes" / "dense_matrix.npy")
        bm25_index = BM25Index.load(root_dir / "indexes" / "bm25.pkl")
        memory_rows = list(read_jsonl(root_dir / "processed" / "memory.jsonl"))
        slot_path = root_dir / "processed" / "memory_slots.jsonl"
        if not slot_path.exists():
            slot_path = root_dir / "processed" / "memory_slots_smoke.jsonl"
        slot_rows = {row["memory_id"]: row for row in read_jsonl(slot_path)} if slot_path.exists() else {}
        slot_blob = np.load(root_dir / "indexes" / "slot_vectors.npz")
        slot_vectors = {key: slot_blob[key] for key in slot_blob.files}
        mem_id_to_idx = {row["memory_id"]: idx for idx, row in enumerate(memory_rows)}
        clusters: dict[str, list[str]] = defaultdict(list)
        mem_id_to_cluster: dict[str, str] = {}
        for row in memory_rows:
            cluster_id = str(row.get("cluster_id") or row["memory_id"])
            clusters[cluster_id].append(row["memory_id"])
            mem_id_to_cluster[row["memory_id"]] = cluster_id
        retriever = cls(
            RetrievalArtifacts(
                dense_index=dense_index,
                bm25_index=bm25_index,
                memory_rows=memory_rows,
                slot_rows=slot_rows,
                slot_vectors=slot_vectors,
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
        retriever.reranker_family = str(cfg_get(cfg, "reranker.family", "slot_bipartite_8x8"))
        retriever.blend_alpha = float(cfg_get(cfg, "reranker.blend_alpha", 0.35))
        feature_model_path = resolve_path(cfg_get(cfg, "reranker.fallback_model_path", "data/V3/models/reranker.pt"))
        retriever.feature_reranker_bundle = load_reranker(feature_model_path)
        if retriever.reranker_family.startswith("slot_bipartite"):
            model_path = resolve_path(cfg_get(cfg, "reranker.model_path", "data/V3/models/slot_bipartite_reranker.pt"))
            retriever.slot_bipartite_bundle = load_slot_bipartite(model_path)
        return retriever

    def retrieve(self, query: str, top_k: int = 5, query_slot_row: dict | None = None) -> tuple[list[RetrieveResult], dict]:
        debug = self.retrieve_debug(query, top_k=top_k, query_slot_row=query_slot_row)
        return debug["all_hits"][: max(1, int(top_k))], debug["trace"]

    def retrieve_debug(self, query: str, top_k: int = 5, query_slot_row: dict | None = None) -> dict:
        candidate_pack = self.collect_candidate_pack(query)
        reranked = self.rerank_candidates(
            query,
            candidate_pack["expanded_ids"],
            candidate_pack["source_map"],
            candidate_pack["coarse_scores"],
            query_slot_row=query_slot_row,
        )
        return {
            "hits": reranked[: max(1, int(top_k))],
            "all_hits": reranked,
            "trace": candidate_pack["trace"],
            "base_ids": candidate_pack["base_ids"],
            "expanded_ids": candidate_pack["expanded_ids"],
            "source_map": candidate_pack["source_map"],
            "coarse_scores": candidate_pack["coarse_scores"],
        }

    def collect_candidate_pack(self, query: str) -> dict:
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

    def rerank_candidates(
        self,
        query: str,
        candidate_ids: list[str],
        source_map: dict[str, str],
        coarse_scores: dict[str, dict[str, float]],
        *,
        query_slot_row: dict | None = None,
    ) -> list[RetrieveResult]:
        if not candidate_ids:
            return []
        query_context = build_query_context(self.artifacts.dense_index, query, slot_row=query_slot_row)
        feature_rows = []
        feature_maps = []
        for mem_id in candidate_ids:
            feature_map = build_candidate_feature_map(
                self.artifacts,
                query_context=query_context,
                mem_id=mem_id,
                source_map=source_map,
                coarse_scores=coarse_scores,
            )
            feature_rows.append(feature_vector_from_map(feature_map))
            feature_maps.append(feature_map)
        feature_matrix = np.asarray(feature_rows, dtype=np.float32)
        support_scores = None
        if self.feature_reranker_bundle is not None and list(self.feature_reranker_bundle.feature_names) == list(FEATURE_NAMES):
            support_scores = score_feature_matrix(self.feature_reranker_bundle, feature_matrix)
        else:
            support_scores = np.asarray([heuristic_score(item) for item in feature_maps], dtype=np.float32)
        if self.slot_bipartite_bundle is not None and self.reranker_family.startswith("slot_bipartite"):
            q_seq = np.stack([query_context["query_slot_vectors"][field] for field in SLOT_FIELDS], axis=0).astype(np.float32)
            q_mask = (np.linalg.norm(q_seq, axis=-1) > 1e-8).astype(bool)
            if not q_mask.any():
                q_mask[0] = True
            mem_seq_batch = np.stack(
                [
                    np.stack(
                        [self.artifacts.slot_vectors[field][self.artifacts.mem_id_to_idx[mem_id]] for field in SLOT_FIELDS],
                        axis=0,
                    )
                    for mem_id in candidate_ids
                ],
                axis=0,
            ).astype(np.float32)
            mem_mask_batch = (np.linalg.norm(mem_seq_batch, axis=-1) > 1e-8).astype(bool)
            empty_rows = ~mem_mask_batch.any(axis=1)
            if np.any(empty_rows):
                mem_mask_batch[empty_rows, 0] = True
            scores = score_many_slot_bipartite(
                self.slot_bipartite_bundle,
                q_seq=q_seq,
                q_mask=q_mask,
                mem_seqs=mem_seq_batch,
                mem_masks=mem_mask_batch,
            )
            scores = scores + self.blend_alpha * support_scores
        elif self.feature_reranker_bundle is not None and list(self.feature_reranker_bundle.feature_names) == list(FEATURE_NAMES):
            scores = support_scores
        else:
            scores = support_scores
        results: list[RetrieveResult] = []
        for mem_id, score in zip(candidate_ids, scores):
            idx = self.artifacts.mem_id_to_idx[mem_id]
            row = self.artifacts.memory_rows[idx]
            slot_row = self.artifacts.slot_rows.get(mem_id, {})
            results.append(
                RetrieveResult(
                    memory_id=mem_id,
                    cluster_id=str(row.get("cluster_id") or mem_id),
                    score=float(score),
                    source=source_map.get(mem_id, ""),
                    display_text=str(row.get("raw_text") or row.get("text") or ""),
                    slot_texts=slot_text_payload(slot_row),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

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
