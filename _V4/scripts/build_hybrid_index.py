from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.retrieval.bm25_index import BM25Index
from agentmemory_v3.retrieval.dense_index import DenseIndex
from agentmemory_v3.retrieval.e5_cache import load_id_list, load_manifest, load_slot_npz, resolve_cache_paths
from agentmemory_v3.utils.io import read_jsonl


SLOT_FIELDS = ("raw_brief", "event", "intent", "entities", "status", "emotion", "context", "impact")


def build_hybrid_index(memory_in: Path, slots_in: Path, out_root: Path, *, config_path: str | Path, alias: str = "") -> dict:
    cfg = load_yaml_config(config_path)
    memory_rows = list(read_jsonl(memory_in))
    slot_rows = {row["memory_id"]: row for row in read_jsonl(slots_in)}
    mem_ids = [row["memory_id"] for row in memory_rows]
    cache_dir = resolve_path(cfg_get(cfg, "cache.dir", "data/V4/VectorCacheV4"))
    cache_alias = alias or str(cfg_get(cfg, "cache.alias", "users"))
    cache_paths = resolve_cache_paths(cache_dir, cache_alias)
    if not cache_paths.manifest.exists():
        raise RuntimeError(f"E5 cache manifest not found: {cache_paths.manifest}")
    manifest = load_manifest(cache_paths)
    cached_mem_ids = load_id_list(cache_paths.memory_ids)
    memory_coarse = np.load(cache_paths.memory_coarse)
    if cached_mem_ids != mem_ids:
        mem_row_map = {mem_id: idx for idx, mem_id in enumerate(cached_mem_ids)}
        if any(mem_id not in mem_row_map for mem_id in mem_ids):
            raise RuntimeError("memory cache ids do not match memory.jsonl")
        order = [mem_row_map[mem_id] for mem_id in mem_ids]
        memory_coarse = memory_coarse[order]
    dense_index = DenseIndex.build(
        mem_ids,
        memory_coarse,
        model_name=str(manifest.get("model_name") or cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
        use_e5_prefix=bool(manifest.get("use_e5_prefix", cfg_get(cfg, "encoder.use_e5_prefix", True))),
        local_files_only=bool(manifest.get("local_files_only", cfg_get(cfg, "encoder.local_files_only", True))),
        offline=bool(manifest.get("offline", cfg_get(cfg, "encoder.offline", True))),
        device=str(cfg_get(cfg, "encoder.device", "auto")),
        batch_size=int(cfg_get(cfg, "encoder.batch_size_cuda", cfg_get(cfg, "encoder.batch_size", 128))),
    )
    search_texts = []
    for row in memory_rows:
        slot = slot_rows.get(row["memory_id"], {})
        parts = [
            str(row.get("raw_text") or ""),
            str(slot.get("raw_brief") or ""),
            str(slot.get("event") or ""),
            str(slot.get("intent") or ""),
            " ".join(str(x) for x in (slot.get("entities") or [])),
            str(slot.get("status") or ""),
            str(slot.get("emotion") or ""),
            str(slot.get("context") or ""),
            str(slot.get("impact") or ""),
        ]
        search_texts.append(" ".join(part for part in parts if part))
    bm25_index = BM25Index.build(mem_ids, search_texts)
    indexes_dir = out_root / "indexes"
    dense_index.save(indexes_dir / "dense_artifact.pkl", indexes_dir / "dense_matrix.npy")
    bm25_index.save(indexes_dir / "bm25.pkl")

    slot_vectors = load_slot_npz(cache_paths.memory_slots)
    if cached_mem_ids != mem_ids:
        mem_row_map = {mem_id: idx for idx, mem_id in enumerate(cached_mem_ids)}
        order = [mem_row_map[mem_id] for mem_id in mem_ids]
        slot_vectors = {field: matrix[order] for field, matrix in slot_vectors.items()}
    np.savez(indexes_dir / "slot_vectors.npz", **slot_vectors)
    return {"memory_count": len(memory_rows), "indexes_dir": str(indexes_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V4 hybrid index.")
    parser.add_argument("--config", default="_V4/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/V4/processed/memory.jsonl")
    parser.add_argument("--slots-in", default="data/V4/processed/memory_slots.jsonl")
    parser.add_argument("--out-root", default="data/V4")
    parser.add_argument("--alias", default="")
    args = parser.parse_args()
    result = build_hybrid_index(
        resolve_path(args.memory_in),
        resolve_path(args.slots_in),
        resolve_path(args.out_root),
        config_path=args.config,
        alias=args.alias,
    )
    print(f"[v4] hybrid index memory={result['memory_count']} -> {result['indexes_dir']}")


if __name__ == "__main__":
    main()
