"""Build user-turn memory files from DeepSeek-style conversation export.

Outputs:
- data/Processed/user_turns_raw.jsonl
- data/Processed/user_turns_dedup.jsonl

Design note:
- Any turn in the same cluster can be treated as a positive/candidate for retrieval.
- Any single turn can be used as query.
- Query paraphrase augmentation should be added before training for robustness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chat_memory_processor import (  # noqa: E402
    ProcessorConfig,
    build_processed_turns,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed user-turn jsonl files.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "RawDeepseekChats" / "conversations.json"),
        help="Input DeepSeek conversation export json.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional fixed session id. Empty means use source conversation id.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.35,
        help="Topic-break similarity threshold (lower => easier to split).",
    )
    parser.add_argument(
        "--raw-out",
        default=str(REPO_ROOT / "data" / "Processed" / "user_turns_raw.jsonl"),
    )
    parser.add_argument(
        "--dedup-out",
        default=str(REPO_ROOT / "data" / "Processed" / "user_turns_dedup.jsonl"),
    )
    args = parser.parse_args()

    raw_rows, dedup_rows = build_processed_turns(
        Path(args.input),
        config=ProcessorConfig(
            session_id=args.session_id or None,
            sim_threshold=args.sim_threshold,
        ),
    )

    raw_out = Path(args.raw_out)
    dedup_out = Path(args.dedup_out)
    write_jsonl(raw_out, raw_rows)
    write_jsonl(dedup_out, dedup_rows)

    excluded = sum(1 for row in raw_rows if row.get("is_excluded"))
    print(f"raw_rows={len(raw_rows)} excluded={excluded} => {raw_out}")
    print(f"dedup_rows={len(dedup_rows)} => {dedup_out}")


if __name__ == "__main__":
    main()
