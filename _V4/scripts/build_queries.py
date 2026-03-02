from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from build_dataset import build_dataset
from agentmemory_v3.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh V4 queries by rebuilding normalized dataset.")
    parser.add_argument("--config", default="_V4/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--out-root", default="data/V4")
    args = parser.parse_args()
    result = build_dataset(
        resolve_path(args.memory_in),
        resolve_path(args.eval_in),
        resolve_path(args.out_root),
    )
    print(f"[v4] queries refreshed -> {result['processed_dir']}")


if __name__ == "__main__":
    main()
