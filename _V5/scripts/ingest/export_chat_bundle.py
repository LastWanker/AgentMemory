from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import resolve_path
from agentmemory_v3.utils.io import read_jsonl, write_jsonl


def export_chat_bundle(memory_in: Path, out_path: Path) -> dict:
    memory_rows = list(read_jsonl(memory_in))
    out_rows = []
    for row in memory_rows:
        search_text = str(row.get("raw_text") or "").strip()
        out_rows.append(
            {
                "memory_id": row["memory_id"],
                "cluster_id": row["cluster_id"],
                "raw_text": row["raw_text"],
                "display_text": row["raw_text"],
                "search_text": search_text,
            }
        )
    write_jsonl(out_path, out_rows)
    return {"count": len(out_rows), "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V5 chat bundle.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/V5/processed/memory.jsonl")
    parser.add_argument("--out", default="data/V5/exports/chat_memory_bundle.jsonl")
    args = parser.parse_args()
    result = export_chat_bundle(resolve_path(args.memory_in), resolve_path(args.out))
    print(f"[v5] chat bundle count={result['count']} -> {result['out']}")


if __name__ == "__main__":
    main()
