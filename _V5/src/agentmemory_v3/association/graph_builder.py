from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from agentmemory_v3.association.llm import AssociationLLM
from agentmemory_v3.encoder import HFSentenceEncoder
from agentmemory_v3.utils.io import ensure_parent, read_jsonl, write_jsonl
from agentmemory_v3.utils.text import unique_keep_order


@dataclass(frozen=True)
class AssociationBuildConfig:
    output_dir: Path
    merge_threshold: float = 0.96
    max_l1_per_memory: int = 10
    l2_target_min: int = 64
    l2_target_max: int = 3000
    l3_target_min: int = 16
    l3_target_max: int = 500
    label_batch_size: int = 256
    bridge_top_k_per_node: int = 20
    bridge_keep_ratio: float = 0.2
    bridge_min_weight: float = 0.18
    bridge_sem_weight: float = 0.3
    bridge_co_weight: float = 0.7
    progress_every: int = 200
    llm_workers: int = 16
    kmeans_use_minibatch: bool = True
    kmeans_minibatch_threshold: int = 12000
    kmeans_batch_size: int = 2048
    kmeans_max_iter: int = 120
    kmeans_random_state: int = 42
    cache_enabled: bool = True
    cache_dir_name: str = "cache"
    llm_l1_cache_file: str = "l1_extract_cache.jsonl"
    llm_parent_seed_cache_file: str = "parent_seed_cache.jsonl"
    llm_cache_version: str = "v1"
    passage_embedding_manifest_file: str = "passage_embedding_manifest.json"
    passage_embedding_matrix_file: str = "passage_embedding_matrix.npy"


class AssociationGraphBuilder:
    def __init__(self, *, encoder: HFSentenceEncoder, llm: AssociationLLM, cfg: AssociationBuildConfig) -> None:
        self.encoder = encoder
        self.llm = llm
        self.cfg = cfg
        self._l1_llm_cache: dict[str, dict] | None = None
        self._l1_llm_cache_dirty = False
        self._parent_seed_cache: dict[str, dict] | None = None
        self._parent_seed_cache_dirty = False
        self._passage_embedding_cache: dict | None = None
        self._passage_embedding_cache_dirty = False

    def _cache_dir(self) -> Path:
        path = self.cfg.output_dir / str(self.cfg.cache_dir_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_l1_llm_cache(self) -> dict[str, dict]:
        if self._l1_llm_cache is not None:
            return self._l1_llm_cache
        cache: dict[str, dict] = {}
        if bool(self.cfg.cache_enabled):
            path = self._cache_dir() / str(self.cfg.llm_l1_cache_file)
            if path.exists():
                for row in read_jsonl(path):
                    memory_id = str(row.get("memory_id") or "").strip()
                    if memory_id:
                        cache[memory_id] = row
        self._l1_llm_cache = cache
        return cache

    def _save_l1_llm_cache(self) -> None:
        if not bool(self.cfg.cache_enabled):
            return
        if not self._l1_llm_cache_dirty or self._l1_llm_cache is None:
            return
        path = self._cache_dir() / str(self.cfg.llm_l1_cache_file)
        rows = [self._l1_llm_cache[k] for k in sorted(self._l1_llm_cache.keys())]
        write_jsonl(path, rows)
        self._l1_llm_cache_dirty = False

    def _load_parent_seed_cache(self) -> dict[str, dict]:
        if self._parent_seed_cache is not None:
            return self._parent_seed_cache
        cache: dict[str, dict] = {}
        if bool(self.cfg.cache_enabled):
            path = self._cache_dir() / str(self.cfg.llm_parent_seed_cache_file)
            if path.exists():
                for row in read_jsonl(path):
                    key = str(row.get("key") or "").strip()
                    if key:
                        cache[key] = row
        self._parent_seed_cache = cache
        return cache

    def _save_parent_seed_cache(self) -> None:
        if not bool(self.cfg.cache_enabled):
            return
        if not self._parent_seed_cache_dirty or self._parent_seed_cache is None:
            return
        path = self._cache_dir() / str(self.cfg.llm_parent_seed_cache_file)
        rows = [self._parent_seed_cache[k] for k in sorted(self._parent_seed_cache.keys())]
        write_jsonl(path, rows)
        self._parent_seed_cache_dirty = False

    def _load_passage_embedding_cache(self) -> dict:
        if self._passage_embedding_cache is not None:
            return self._passage_embedding_cache
        empty_cache = {
            "texts": [],
            "text_to_idx": {},
            "matrix": np.zeros((0, self.encoder.dim), dtype=np.float32),
        }
        if not bool(self.cfg.cache_enabled):
            self._passage_embedding_cache = empty_cache
            return empty_cache

        manifest_path = self._cache_dir() / str(self.cfg.passage_embedding_manifest_file)
        matrix_path = self._cache_dir() / str(self.cfg.passage_embedding_matrix_file)
        if not manifest_path.exists() or not matrix_path.exists():
            self._passage_embedding_cache = empty_cache
            return empty_cache

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            texts = [str(item) for item in manifest.get("texts") or []]
            matrix = np.load(matrix_path).astype(np.float32)
            valid = (
                str(manifest.get("model_name") or "") == str(self.encoder.config.model_name)
                and bool(manifest.get("use_e5_prefix", True)) == bool(self.encoder.config.use_e5_prefix)
                and int(matrix.shape[1]) == int(self.encoder.dim)
                and int(matrix.shape[0]) == len(texts)
            )
            if not valid:
                self._passage_embedding_cache = empty_cache
                return empty_cache
            text_to_idx = {text: idx for idx, text in enumerate(texts)}
            self._passage_embedding_cache = {"texts": texts, "text_to_idx": text_to_idx, "matrix": matrix}
            return self._passage_embedding_cache
        except Exception:
            self._passage_embedding_cache = empty_cache
            return empty_cache

    def _save_passage_embedding_cache(self) -> None:
        if not bool(self.cfg.cache_enabled):
            return
        if not self._passage_embedding_cache_dirty or self._passage_embedding_cache is None:
            return
        cache = self._passage_embedding_cache
        manifest_path = self._cache_dir() / str(self.cfg.passage_embedding_manifest_file)
        matrix_path = self._cache_dir() / str(self.cfg.passage_embedding_matrix_file)
        manifest = {
            "version": "v1",
            "model_name": str(self.encoder.config.model_name),
            "use_e5_prefix": bool(self.encoder.config.use_e5_prefix),
            "dim": int(self.encoder.dim),
            "texts": cache["texts"],
        }
        ensure_parent(manifest_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(matrix_path, np.asarray(cache["matrix"], dtype=np.float32))
        self._passage_embedding_cache_dirty = False

    def _encode_passage_texts_cached(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)
        if not bool(self.cfg.cache_enabled):
            return self.encoder.encode_passage_texts(texts)

        cache = self._load_passage_embedding_cache()
        unique_texts = unique_keep_order([str(item).strip() for item in texts if str(item).strip()])
        if not unique_texts:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)

        missing: list[str] = [text for text in unique_texts if text not in cache["text_to_idx"]]
        hit_count = len(unique_texts) - len(missing)
        if missing:
            miss_matrix = self.encoder.encode_passage_texts(missing)
            base = int(cache["matrix"].shape[0])
            cache["matrix"] = (
                np.vstack([cache["matrix"], miss_matrix]).astype(np.float32)
                if cache["matrix"].size
                else miss_matrix.astype(np.float32)
            )
            for offset, text in enumerate(missing):
                idx = base + offset
                cache["texts"].append(text)
                cache["text_to_idx"][text] = idx
            self._passage_embedding_cache_dirty = True
            self._save_passage_embedding_cache()
        print(f"[assoc][cache] passage_embeddings unique={len(unique_texts)} hit={hit_count} miss={len(missing)}")

        out = np.zeros((len(texts), self.encoder.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            key = str(text).strip()
            vec_idx = cache["text_to_idx"].get(key)
            if vec_idx is not None:
                out[idx] = cache["matrix"][int(vec_idx)]
        return out

    def build_and_save(self, memory_rows: list[dict]) -> dict:
        l1_bundle = self._build_l1(memory_rows)
        l2_bundle = self._build_parent_level(
            child_names=l1_bundle["names"],
            target_level="L2",
            child_vectors=l1_bundle["vectors"],
            target_count=_target_count(
                len(l1_bundle["names"]),
                minimum=self.cfg.l2_target_min,
                maximum=self.cfg.l2_target_max,
                divisor=6,
            ),
            id_prefix="l2",
        )
        l3_bundle = self._build_parent_level(
            child_names=l2_bundle["names"],
            target_level="L3",
            child_vectors=l2_bundle["vectors"],
            target_count=_target_count(
                len(l2_bundle["names"]),
                minimum=self.cfg.l3_target_min,
                maximum=self.cfg.l3_target_max,
                divisor=8,
            ),
            id_prefix="l3",
        )

        parent_edges_l1_l2 = _build_parent_edges(
            child_ids=l1_bundle["ids"],
            parent_ids=l2_bundle["ids"],
            parent_indices=l2_bundle["assignments"]["parent_indices"],
            confidences=l2_bundle["assignments"]["confidences"],
        )
        parent_edges_l2_l3 = _build_parent_edges(
            child_ids=l2_bundle["ids"],
            parent_ids=l3_bundle["ids"],
            parent_indices=l3_bundle["assignments"]["parent_indices"],
            confidences=l3_bundle["assignments"]["confidences"],
        )
        parent_edges = parent_edges_l1_l2 + parent_edges_l2_l3

        bridge_edges = self._build_l1_bridge_edges(
            l1_ids=l1_bundle["ids"],
            l1_vectors=l1_bundle["vectors"],
            memory_links=l1_bundle["memory_links"],
        )

        concepts = []
        for node_id, name, aliases in zip(l1_bundle["ids"], l1_bundle["names"], l1_bundle["aliases"]):
            concepts.append(
                {
                    "node_id": node_id,
                    "level": "L1",
                    "canonical_name": name,
                    "aliases": sorted(aliases),
                }
            )
        for node_id, name in zip(l2_bundle["ids"], l2_bundle["names"]):
            concepts.append(
                {
                    "node_id": node_id,
                    "level": "L2",
                    "canonical_name": name,
                    "aliases": [],
                }
            )
        for node_id, name in zip(l3_bundle["ids"], l3_bundle["names"]):
            concepts.append(
                {
                    "node_id": node_id,
                    "level": "L3",
                    "canonical_name": name,
                    "aliases": [],
                }
            )

        concept_ids = l1_bundle["ids"] + l2_bundle["ids"] + l3_bundle["ids"]
        concept_matrix = np.vstack([l1_bundle["vectors"], l2_bundle["vectors"], l3_bundle["vectors"]]).astype(np.float32)
        l1_inverted = _build_l1_inverted(l1_bundle["memory_links"])

        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "concepts.jsonl", concepts)
        write_jsonl(out / "parent_edges.jsonl", parent_edges)
        write_jsonl(out / "bridge_edges.jsonl", bridge_edges)
        write_jsonl(out / "memory_links.jsonl", _memory_links_to_rows(l1_bundle["memory_links"]))
        _write_json(out / "l1_inverted.json", l1_inverted)
        _write_json(out / "concept_ids.json", concept_ids)
        _write_json(out / "l1_ids.json", l1_bundle["ids"])
        np.save(out / "concept_matrix.npy", concept_matrix)
        np.save(out / "l1_matrix.npy", l1_bundle["vectors"].astype(np.float32))

        manifest = {
            "version": "v5_association_v1",
            "memory_count": len(memory_rows),
            "l1_count": len(l1_bundle["ids"]),
            "l2_count": len(l2_bundle["ids"]),
            "l3_count": len(l3_bundle["ids"]),
            "parent_edge_count": len(parent_edges),
            "bridge_edge_count": len(bridge_edges),
            "merge_threshold": float(self.cfg.merge_threshold),
            "max_l1_per_memory": int(self.cfg.max_l1_per_memory),
            "encoder_model": self.encoder.config.model_name,
            "parent_assignment": "seeded_kmeans",
        }
        _write_json(out / "manifest.json", manifest)
        return manifest

    def _build_l1(self, memory_rows: list[dict]) -> dict:
        memory_candidates: dict[str, list[str]] = {}
        total = len(memory_rows)
        progress_every = max(0, int(self.cfg.progress_every))
        llm_workers = max(1, int(self.cfg.llm_workers))
        l1_cache = self._load_l1_llm_cache()
        cache_hits = 0
        cache_misses = 0

        def _cache_hit(memory_id: str, raw_text: str) -> list[str] | None:
            row = l1_cache.get(memory_id)
            if not row:
                return None
            if str(row.get("cache_version") or "") != str(self.cfg.llm_cache_version):
                return None
            if str(row.get("raw_hash") or "") != _hash_text(raw_text):
                return None
            if int(row.get("max_items") or 0) != int(self.cfg.max_l1_per_memory):
                return None
            values = row.get("concepts")
            if not isinstance(values, list):
                return None
            return unique_keep_order([str(item).strip() for item in values if str(item).strip()])[: self.cfg.max_l1_per_memory]

        if llm_workers <= 1:
            for idx, row in enumerate(memory_rows, start=1):
                memory_id = str(row.get("memory_id") or "").strip()
                if not memory_id:
                    continue
                raw_text = str(row.get("raw_text") or row.get("text") or "").strip()
                cached_values = _cache_hit(memory_id, raw_text)
                if cached_values is not None:
                    concepts = cached_values
                    cache_hits += 1
                else:
                    concepts = self.llm.extract_l1_concepts(raw_text, max_items=self.cfg.max_l1_per_memory)
                    if not concepts:
                        concepts = [raw_text[:24]] if raw_text else []
                    concepts = unique_keep_order([c for c in concepts if c])[: self.cfg.max_l1_per_memory]
                    l1_cache[memory_id] = {
                        "memory_id": memory_id,
                        "raw_hash": _hash_text(raw_text),
                        "max_items": int(self.cfg.max_l1_per_memory),
                        "cache_version": str(self.cfg.llm_cache_version),
                        "llm_model": str(self.llm.cfg.model),
                        "concepts": concepts,
                    }
                    self._l1_llm_cache_dirty = True
                    cache_misses += 1
                memory_candidates[memory_id] = concepts
                if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
                    print(f"[assoc][build] L1 extract {idx}/{total}")
        else:
            jobs: list[tuple[str, str, str]] = []
            for row in memory_rows:
                memory_id = str(row.get("memory_id") or "").strip()
                if not memory_id:
                    continue
                raw_text = str(row.get("raw_text") or row.get("text") or "").strip()
                cached_values = _cache_hit(memory_id, raw_text)
                if cached_values is not None:
                    memory_candidates[memory_id] = cached_values
                    cache_hits += 1
                    continue
                jobs.append((memory_id, raw_text, _hash_text(raw_text)))

            thread_state = threading.local()

            def _extract(raw_text: str) -> list[str]:
                llm_local = getattr(thread_state, "llm", None)
                if llm_local is None:
                    llm_local = AssociationLLM(self.llm.cfg)
                    thread_state.llm = llm_local
                concepts_local = llm_local.extract_l1_concepts(raw_text, max_items=self.cfg.max_l1_per_memory)
                if not concepts_local:
                    concepts_local = [raw_text[:24]] if raw_text else []
                return unique_keep_order([c for c in concepts_local if c])[: self.cfg.max_l1_per_memory]

            completed = 0
            total_jobs = len(jobs)
            output_by_idx: dict[int, list[str]] = {}
            with ThreadPoolExecutor(max_workers=llm_workers) as executor:
                future_to_idx = {
                    executor.submit(_extract, raw_text): idx for idx, (_, raw_text, _) in enumerate(jobs)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    output_by_idx[idx] = future.result()
                    completed += 1
                    if progress_every > 0 and (
                        completed == 1 or completed % progress_every == 0 or completed == total_jobs
                    ):
                        print(f"[assoc][build] L1 extract {completed}/{total_jobs}")

            for idx, (memory_id, _, raw_hash) in enumerate(jobs):
                values = output_by_idx.get(idx, [])
                memory_candidates[memory_id] = values
                l1_cache[memory_id] = {
                    "memory_id": memory_id,
                    "raw_hash": raw_hash,
                    "max_items": int(self.cfg.max_l1_per_memory),
                    "cache_version": str(self.cfg.llm_cache_version),
                    "llm_model": str(self.llm.cfg.model),
                    "concepts": values,
                }
                self._l1_llm_cache_dirty = True
                cache_misses += 1

        self._save_l1_llm_cache()
        print(f"[assoc][cache] l1_llm hit={cache_hits} miss={cache_misses}")

        name_stream: list[str] = []
        for row in memory_rows:
            memory_id = str(row.get("memory_id") or "").strip()
            if not memory_id:
                continue
            name_stream.extend(memory_candidates.get(memory_id, []))

        unique_names = unique_keep_order(name_stream)
        if not unique_names:
            return {
                "ids": [],
                "names": [],
                "aliases": [],
                "vectors": np.zeros((0, self.encoder.dim), dtype=np.float32),
                "memory_links": {},
            }

        unique_vectors = self._encode_passage_texts_cached(unique_names)
        merge = _merge_names_by_similarity(
            names=unique_names,
            vectors=unique_vectors,
            threshold=self.cfg.merge_threshold,
            id_prefix="l1",
        )

        name_to_l1_id = merge["name_to_node_id"]
        memory_links: dict[str, list[str]] = {}
        for memory_id, names in memory_candidates.items():
            mapped = [name_to_l1_id[name] for name in names if name in name_to_l1_id]
            memory_links[memory_id] = unique_keep_order(mapped)

        return {
            "ids": merge["node_ids"],
            "names": merge["canonical_names"],
            "aliases": merge["aliases"],
            "vectors": merge["canonical_vectors"],
            "memory_links": memory_links,
        }

    def _build_parent_level(
        self,
        *,
        child_names: list[str],
        target_level: str,
        child_vectors: np.ndarray,
        target_count: int,
        id_prefix: str,
    ) -> dict:
        if not child_names:
            return {
                "ids": [],
                "names": [],
                "vectors": np.zeros((0, self.encoder.dim), dtype=np.float32),
                "assignments": {"parent_indices": np.zeros((0,), dtype=np.int32), "confidences": np.zeros((0,), dtype=np.float32)},
            }
        seed_names = self._build_parent_seeds(child_names, target_level=target_level, target_count=target_count)
        if not seed_names:
            seed_names = child_names[: max(1, min(target_count, len(child_names)))]
        seed_vectors = self._encode_passage_texts_cached(seed_names)
        merged = _merge_names_by_similarity(
            names=seed_names,
            vectors=seed_vectors,
            threshold=self.cfg.merge_threshold,
            id_prefix=id_prefix,
        )
        parent_ids = merged["node_ids"]
        parent_names = merged["canonical_names"]
        parent_vectors = merged["canonical_vectors"]
        if parent_vectors.shape[0] == 0:
            parent_ids = [f"{id_prefix}_000001"]
            parent_names = [f"{target_level.lower()}_default"]
            parent_vectors = np.mean(child_vectors, axis=0, keepdims=True).astype(np.float32)
            parent_vectors = _l2_normalize(parent_vectors)
        max_parent_count = max(1, min(int(parent_vectors.shape[0]), int(child_vectors.shape[0])))
        parent_ids = parent_ids[:max_parent_count]
        parent_names = parent_names[:max_parent_count]
        parent_vectors = parent_vectors[:max_parent_count]
        clustered = _seeded_kmeans_assign(
            child_vectors=child_vectors,
            seed_vectors=parent_vectors,
            use_minibatch=bool(self.cfg.kmeans_use_minibatch),
            minibatch_threshold=int(self.cfg.kmeans_minibatch_threshold),
            batch_size=int(self.cfg.kmeans_batch_size),
            max_iter=int(self.cfg.kmeans_max_iter),
            random_state=int(self.cfg.kmeans_random_state),
        )
        kept_seed_indices = clustered["kept_seed_indices"]
        parent_ids = [parent_ids[idx] for idx in kept_seed_indices]
        parent_names = [parent_names[idx] for idx in kept_seed_indices]
        parent_vectors = clustered["parent_vectors"]
        assignments = {
            "parent_indices": clustered["parent_indices"],
            "confidences": clustered["confidences"],
        }
        return {
            "ids": parent_ids,
            "names": parent_names,
            "vectors": parent_vectors,
            "assignments": assignments,
        }

    def _build_parent_seeds(self, child_names: list[str], *, target_level: str, target_count: int) -> list[str]:
        if not child_names:
            return []
        cache_key_payload = {
            "cache_version": str(self.cfg.llm_cache_version),
            "target_level": str(target_level),
            "target_count": int(target_count),
            "child_names": [str(item) for item in child_names],
            "label_batch_size": int(self.cfg.label_batch_size),
            "llm_model": str(self.llm.cfg.model),
        }
        cache_key = _hash_text(json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True))
        seed_cache = self._load_parent_seed_cache()
        cached = seed_cache.get(cache_key)
        if cached and isinstance(cached.get("labels"), list):
            labels_cached = unique_keep_order([str(item).strip() for item in cached.get("labels") if str(item).strip()])
            print(f"[assoc][cache] parent_seed hit level={target_level} labels={len(labels_cached)}")
            return labels_cached[: max(1, int(target_count))]

        labels: list[str] = []
        batch_size = max(8, int(self.cfg.label_batch_size))
        total = len(child_names)
        for start in range(0, total, batch_size):
            batch = child_names[start : start + batch_size]
            per_batch_target = max(1, int(round(target_count * len(batch) / max(total, 1))))
            labels.extend(self.llm.propose_parent_labels(batch, target_level=target_level, target_count=per_batch_target))
            end = min(total, start + batch_size)
            print(f"[assoc][build] {target_level} seed naming {end}/{total}")
        labels = unique_keep_order([item for item in labels if item])
        if len(labels) > target_count:
            labels = labels[:target_count]
        seed_cache[cache_key] = {
            "key": cache_key,
            "target_level": str(target_level),
            "target_count": int(target_count),
            "cache_version": str(self.cfg.llm_cache_version),
            "llm_model": str(self.llm.cfg.model),
            "labels": labels,
        }
        self._parent_seed_cache_dirty = True
        self._save_parent_seed_cache()
        return labels

    def _build_l1_bridge_edges(
        self,
        *,
        l1_ids: list[str],
        l1_vectors: np.ndarray,
        memory_links: dict[str, list[str]],
    ) -> list[dict]:
        if not l1_ids:
            return []
        id_to_idx = {node_id: idx for idx, node_id in enumerate(l1_ids)}
        freq: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        for l1_list in memory_links.values():
            idxs = sorted({id_to_idx[item] for item in l1_list if item in id_to_idx})
            for idx in idxs:
                freq[idx] += 1
            for a, b in combinations(idxs, 2):
                pair_counts[(a, b)] += 1
        if not pair_counts:
            return []

        raw_edges: list[tuple[int, int, float]] = []
        for (a, b), count in pair_counts.items():
            co_norm = float(count) / max(1e-6, math.sqrt(float(freq[a]) * float(freq[b])))
            sem = float(np.dot(l1_vectors[a], l1_vectors[b]))
            weight = float(self.cfg.bridge_co_weight) * co_norm + float(self.cfg.bridge_sem_weight) * max(0.0, sem)
            if weight < float(self.cfg.bridge_min_weight):
                continue
            raw_edges.append((a, b, weight))
        if not raw_edges:
            return []

        top_k = max(1, int(self.cfg.bridge_top_k_per_node))
        per_node: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        for edge in raw_edges:
            a, b, _ = edge
            per_node[a].append(edge)
            per_node[b].append(edge)
        kept_pairs: set[tuple[int, int]] = set()
        for idx, edges in per_node.items():
            del idx
            edges_sorted = sorted(edges, key=lambda item: item[2], reverse=True)[:top_k]
            for a, b, _ in edges_sorted:
                kept_pairs.add((a, b) if a < b else (b, a))
        filtered = [edge for edge in raw_edges if ((edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])) in kept_pairs]
        filtered = sorted(filtered, key=lambda item: item[2], reverse=True)
        keep_ratio = float(self.cfg.bridge_keep_ratio)
        keep_n = max(1, int(round(len(filtered) * keep_ratio)))
        filtered = filtered[:keep_n]

        out = []
        for a, b, weight in filtered:
            out.append(
                {
                    "source_id": l1_ids[a],
                    "target_id": l1_ids[b],
                    "weight": float(weight),
                    "kind": "l1_bridge",
                }
            )
        return out


def _merge_names_by_similarity(
    *,
    names: list[str],
    vectors: np.ndarray,
    threshold: float,
    id_prefix: str,
) -> dict:
    canonical_names: list[str] = []
    canonical_vecs: list[np.ndarray] = []
    aliases: list[set[str]] = []
    name_to_idx: dict[str, int] = {}
    canonical_matrix: np.ndarray | None = None
    for name, vec in zip(names, vectors):
        if name in name_to_idx:
            aliases[name_to_idx[name]].add(name)
            continue
        idx = -1
        if canonical_vecs:
            if canonical_matrix is None:
                canonical_matrix = np.vstack(canonical_vecs).astype(np.float32)
            sims = canonical_matrix @ vec
            best = int(np.argmax(sims))
            if float(sims[best]) >= float(threshold):
                idx = best
        if idx < 0:
            idx = len(canonical_names)
            canonical_names.append(name)
            canonical_vecs.append(np.asarray(vec, dtype=np.float32))
            aliases.append({name})
            canonical_matrix = None
        else:
            aliases[idx].add(name)
        name_to_idx[name] = idx
    node_ids = [f"{id_prefix}_{idx + 1:06d}" for idx in range(len(canonical_names))]
    name_to_node_id = {name: node_ids[idx] for name, idx in name_to_idx.items()}
    canonical_vectors = np.vstack(canonical_vecs).astype(np.float32) if canonical_vecs else np.zeros((0, 0), dtype=np.float32)
    canonical_vectors = _l2_normalize(canonical_vectors)
    return {
        "node_ids": node_ids,
        "canonical_names": canonical_names,
        "canonical_vectors": canonical_vectors,
        "aliases": aliases,
        "name_to_node_id": name_to_node_id,
    }


def _seeded_kmeans_assign(
    *,
    child_vectors: np.ndarray,
    seed_vectors: np.ndarray,
    use_minibatch: bool,
    minibatch_threshold: int,
    batch_size: int,
    max_iter: int,
    random_state: int,
) -> dict:
    n_children = int(child_vectors.shape[0])
    n_parents = int(seed_vectors.shape[0])
    parent_indices = np.zeros((n_children,), dtype=np.int32)
    confidences = np.zeros((n_children,), dtype=np.float32)
    if n_children == 0 or n_parents == 0:
        return {
            "parent_indices": parent_indices,
            "confidences": confidences,
            "parent_vectors": np.zeros((0, seed_vectors.shape[1] if seed_vectors.ndim == 2 else 0), dtype=np.float32),
            "kept_seed_indices": [],
        }

    n_clusters = max(1, min(n_children, n_parents))
    init_centers = np.asarray(seed_vectors[:n_clusters], dtype=np.float32)
    use_mb = bool(use_minibatch and n_children >= max(1, int(minibatch_threshold)))
    if use_mb:
        estimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            batch_size=max(n_clusters, int(batch_size)),
            max_iter=max(10, int(max_iter)),
            random_state=int(random_state),
            reassignment_ratio=0.0,
        )
    else:
        estimator = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            max_iter=max(10, int(max_iter)),
            random_state=int(random_state),
            algorithm="lloyd",
        )
    estimator.fit(child_vectors)
    parent_indices = np.asarray(estimator.labels_, dtype=np.int32)
    parent_vectors = _l2_normalize(np.asarray(estimator.cluster_centers_, dtype=np.float32))

    used = np.unique(parent_indices)
    remap = np.full((parent_vectors.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.shape[0]), dtype=np.int32)
    parent_indices = remap[parent_indices]
    kept_seed_indices = used.astype(np.int32)
    parent_vectors = parent_vectors[used]

    confidences = np.sum(child_vectors * parent_vectors[parent_indices], axis=1)
    confidences = np.clip(confidences, 0.0, 1.0).astype(np.float32)
    return {
        "parent_indices": parent_indices,
        "confidences": confidences,
        "parent_vectors": parent_vectors,
        "kept_seed_indices": [int(x) for x in kept_seed_indices.tolist()],
    }


def _build_parent_edges(
    *,
    child_ids: list[str],
    parent_ids: list[str],
    parent_indices: np.ndarray,
    confidences: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    for child_idx, child_id in enumerate(child_ids):
        if child_idx >= parent_indices.shape[0]:
            continue
        parent_idx = int(parent_indices[child_idx])
        if parent_idx < 0 or parent_idx >= len(parent_ids):
            continue
        out.append(
            {
                "child_id": child_id,
                "parent_id": parent_ids[parent_idx],
                "conf": float(confidences[child_idx]),
            }
        )
    return out


def _build_l1_inverted(memory_links: dict[str, list[str]]) -> dict[str, list[str]]:
    inverted: dict[str, list[str]] = defaultdict(list)
    for memory_id, l1_ids in memory_links.items():
        for l1_id in l1_ids:
            inverted[l1_id].append(memory_id)
    return {k: v for k, v in inverted.items()}


def _memory_links_to_rows(memory_links: dict[str, list[str]]) -> list[dict]:
    return [{"memory_id": memory_id, "l1_ids": l1_ids} for memory_id, l1_ids in memory_links.items()]


def _hash_text(text: str) -> str:
    payload = str(text or "").encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _target_count(size: int, *, minimum: int, maximum: int, divisor: int) -> int:
    if size <= 0:
        return minimum
    value = max(minimum, size // max(1, int(divisor)))
    return min(maximum, value)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape((0, 0)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
