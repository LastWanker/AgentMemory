from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from build_dataset import build_dataset
from build_hybrid_index import build_hybrid_index
from build_query_slots import build_query_slots
from build_slots import build_slots
from export_chat_bundle import export_chat_bundle
from agentmemory_v3.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V4 demo assets in one command.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--slots-limit", type=int, default=None)
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--out-root", default="data/V4")
    args = parser.parse_args()

    out_root = resolve_path(args.out_root)
    build_dataset(resolve_path(args.memory_in), resolve_path(args.eval_in), out_root)

    slots_limit = args.slots_limit
    if slots_limit is None:
        slots_limit = 3 if args.profile == "smoke" else 0
    slots_name = "memory_slots_smoke.jsonl" if args.profile == "smoke" else "memory_slots.jsonl"
    slots_path = out_root / "processed" / slots_name
    build_slots(out_root / "processed" / "memory.jsonl", slots_path, limit=int(slots_limit))
    query_slots_limit = 30 if args.profile == "smoke" else 0
    build_query_slots(out_root / "processed" / "query.jsonl", out_root / "processed" / "query_slots.jsonl", limit=int(query_slots_limit))
    build_hybrid_index(out_root / "processed" / "memory.jsonl", slots_path, out_root)
    export_chat_bundle(out_root / "processed" / "memory.jsonl", slots_path, out_root / "exports" / "chat_memory_bundle.jsonl")
    print(f"[v4][runtime] profile={args.profile} ready -> {out_root}")


if __name__ == "__main__":
    main()
