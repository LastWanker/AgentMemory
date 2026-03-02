from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import resolve_path
from agentmemory_v3.utils.io import read_jsonl, write_jsonl


def export_chat_bundle(memory_in: Path, slots_in: Path, out_path: Path) -> dict:
    memory_rows = list(read_jsonl(memory_in))
    slot_rows = {row["memory_id"]: row for row in read_jsonl(slots_in)}
    out_rows = []
    for row in memory_rows:
        slot = slot_rows.get(row["memory_id"], {})
        slot_texts = {
            "raw_brief": slot.get("raw_brief", ""),
            "event": slot.get("event", ""),
            "intent": slot.get("intent", ""),
            "entities": slot.get("entities", []),
            "status": slot.get("status", ""),
            "emotion": slot.get("emotion", ""),
            "context": slot.get("context", ""),
            "impact": slot.get("impact", ""),
        }
        search_text = " ".join(
            [
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
        ).strip()
        out_rows.append(
            {
                "memory_id": row["memory_id"],
                "cluster_id": row["cluster_id"],
                "raw_text": row["raw_text"],
                "display_text": row["raw_text"],
                "search_text": search_text,
                "slot_texts": slot_texts,
            }
        )
    write_jsonl(out_path, out_rows)
    return {"count": len(out_rows), "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V3 chat bundle.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/V3/processed/memory.jsonl")
    parser.add_argument("--slots-in", default="data/V3/processed/memory_slots.jsonl")
    parser.add_argument("--out", default="data/V3/exports/chat_memory_bundle.jsonl")
    args = parser.parse_args()
    result = export_chat_bundle(resolve_path(args.memory_in), resolve_path(args.slots_in), resolve_path(args.out))
    print(f"[v3] chat bundle count={result['count']} -> {result['out']}")


if __name__ == "__main__":
    main()
