from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Doctor V3 data chain.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V3"))
    checks = {
        "memory": root_dir / "processed" / "memory.jsonl",
        "query": root_dir / "processed" / "query.jsonl",
        "query_slots": root_dir / "processed" / "query_slots.jsonl",
        "cluster": root_dir / "processed" / "cluster.jsonl",
        "slots": root_dir / "processed" / "memory_slots.jsonl",
        "dense_artifact": root_dir / "indexes" / "dense_artifact.pkl",
        "dense_matrix": root_dir / "indexes" / "dense_matrix.npy",
        "bm25": root_dir / "indexes" / "bm25.pkl",
        "slot_vectors": root_dir / "indexes" / "slot_vectors.npz",
        "train_samples": root_dir / "training" / "train_samples.jsonl",
        "reranker_model": resolve_path(cfg_get(cfg, "reranker.model_path", "data/V3/models/reranker.pt")),
    }
    report = {}
    for name, path in checks.items():
        report[name] = {"exists": path.exists(), "path": str(path)}
        if path.exists() and path.is_file():
            report[name]["size_bytes"] = path.stat().st_size
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
