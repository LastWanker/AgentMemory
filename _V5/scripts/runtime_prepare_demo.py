from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from build_dataset import build_dataset
from build_e5_cache import build_e5_cache
from build_hybrid_index import build_hybrid_index
from export_chat_bundle import export_chat_bundle
from agentmemory_v3.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V5 coarse-only demo assets in one command.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--out-root", default="data/V5")
    args = parser.parse_args()

    out_root = resolve_path(args.out_root)
    build_dataset(resolve_path(args.memory_in), resolve_path(args.eval_in), out_root)

    build_e5_cache(args.config)
    build_hybrid_index(out_root / "processed" / "memory.jsonl", out_root, config_path=args.config)
    export_chat_bundle(out_root / "processed" / "memory.jsonl", out_root / "exports" / "chat_memory_bundle.jsonl")
    print(f"[v5][runtime] profile={args.profile} ready -> {out_root}")


if __name__ == "__main__":
    main()
