from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.retrieval.e5_cache import resolve_cache_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Doctor V5 coarse-only data chain.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
    cache_dir = resolve_path(cfg_get(cfg, "cache.dir", "data/V5/VectorCacheV5"))
    cache_alias = str(cfg_get(cfg, "cache.alias", "users"))
    cache_paths = resolve_cache_paths(cache_dir, cache_alias)
    checks = {
        "memory": root_dir / "processed" / "memory.jsonl",
        "query": root_dir / "processed" / "query.jsonl",
        "cluster": root_dir / "processed" / "cluster.jsonl",
        "dense_artifact": root_dir / "indexes" / "dense_artifact.pkl",
        "dense_matrix": root_dir / "indexes" / "dense_matrix.npy",
        "bm25": root_dir / "indexes" / "bm25.pkl",
        "chat_bundle": root_dir / "exports" / "chat_memory_bundle.jsonl",
        "cache_manifest": cache_paths.manifest,
        "cache_memory_ids": cache_paths.memory_ids,
        "cache_query_ids": cache_paths.query_ids,
        "cache_memory_coarse": cache_paths.memory_coarse,
        "cache_query_coarse": cache_paths.query_coarse,
    }
    report = {}
    for name, path in checks.items():
        report[name] = {"exists": path.exists(), "path": str(path)}
        if path.exists() and path.is_file():
            report[name]["size_bytes"] = path.stat().st_size
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
