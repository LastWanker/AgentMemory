from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import resolve_path
from agentmemory_v3.retrieval.bm25_index import BM25Index
from agentmemory_v3.retrieval.dense_index import DenseIndex
from agentmemory_v3.utils.io import read_jsonl


SLOT_FIELDS = ("raw_brief", "event", "intent", "entities", "status", "emotion", "context", "impact")


def build_hybrid_index(memory_in: Path, slots_in: Path, out_root: Path) -> dict:
    memory_rows = list(read_jsonl(memory_in))
    slot_rows = {row["memory_id"]: row for row in read_jsonl(slots_in)}
    mem_ids = [row["memory_id"] for row in memory_rows]
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
    dense_index = DenseIndex.build(mem_ids, search_texts)
    bm25_index = BM25Index.build(mem_ids, search_texts)
    indexes_dir = out_root / "indexes"
    dense_index.save(indexes_dir / "dense_artifact.pkl", indexes_dir / "dense_matrix.npy")
    bm25_index.save(indexes_dir / "bm25.pkl")

    slot_vectors = {}
    for field in SLOT_FIELDS:
        texts = []
        for row in memory_rows:
            slot = slot_rows.get(row["memory_id"], {})
            value = slot.get(field, "")
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            texts.append(str(value or ""))
        slot_vectors[field] = dense_index.transform_texts(texts)
    np.savez(indexes_dir / "slot_vectors.npz", **slot_vectors)
    return {"memory_count": len(memory_rows), "indexes_dir": str(indexes_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 hybrid index.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/V3/processed/memory.jsonl")
    parser.add_argument("--slots-in", default="data/V3/processed/memory_slots.jsonl")
    parser.add_argument("--out-root", default="data/V3")
    args = parser.parse_args()
    result = build_hybrid_index(
        resolve_path(args.memory_in),
        resolve_path(args.slots_in),
        resolve_path(args.out_root),
    )
    print(f"[v3] hybrid index memory={result['memory_count']} -> {result['indexes_dir']}")


if __name__ == "__main__":
    main()
