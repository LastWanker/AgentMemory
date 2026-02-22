"""Build chat memory artifacts from DeepSeek conversations export.

Outputs (default):
- data/Processed/user_turns_raw.jsonl
- data/Processed/user_turns_dedup.jsonl
- data/Processed/memory_chat.jsonl

Query/eval datasets should be generated separately via:
- scripts/memory_processer/build_chat_queries.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chat_memory_processor import ProcessorConfig, build_processed_turns, write_jsonl  # noqa: E402


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"YAML config requested but pyyaml is unavailable: {path}") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config root must be object/dict: {path}")
        return payload
    raise ValueError(f"Unsupported config file type: {path}")


def _cfg(data: dict, key: str, default):
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed chat memory datasets.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "chat_memory.yaml"),
        help="Optional JSON/YAML config file.",
    )
    parser.add_argument(
        "--input",
        default=None,
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--segmentation-mode", choices=("adaptive", "fixed"), default=None)
    parser.add_argument(
        "--fixed-sim-threshold",
        type=float,
        default=None,
        help="Used when --segmentation-mode fixed.",
    )
    parser.add_argument("--novelty-threshold", type=float, default=None)
    parser.add_argument("--min-segment-len", type=int, default=None)
    parser.add_argument("--no-cross-session-merge", action="store_true")
    parser.add_argument("--global-merge-sim-threshold", type=float, default=None)
    parser.add_argument(
        "--raw-out",
        default=None,
    )
    parser.add_argument(
        "--dedup-out",
        default=None,
    )
    parser.add_argument(
        "--memory-out",
        default=None,
    )
    args = parser.parse_args()
    loaded = _load_config(Path(args.config))

    input_path = Path(args.input or _cfg(loaded, "input", str(REPO_ROOT / "data" / "RawDeepseekChats" / "conversations.json")))
    session_id = args.session_id or _cfg(loaded, "session_id", "")
    segmentation_mode = args.segmentation_mode or _cfg(loaded, "segmentation.mode", "adaptive")
    fixed_sim_threshold = (
        args.fixed_sim_threshold
        if args.fixed_sim_threshold is not None
        else float(_cfg(loaded, "segmentation.fixed_sim_threshold", 0.35))
    )
    novelty_threshold = (
        args.novelty_threshold
        if args.novelty_threshold is not None
        else float(_cfg(loaded, "segmentation.novelty_threshold", 0.80))
    )
    min_segment_len = (
        args.min_segment_len
        if args.min_segment_len is not None
        else int(_cfg(loaded, "segmentation.min_segment_len", 2))
    )
    cross_session_merge = (
        not args.no_cross_session_merge
        if args.no_cross_session_merge
        else bool(_cfg(loaded, "clustering.cross_session_merge", True))
    )
    global_merge_sim_threshold = (
        args.global_merge_sim_threshold
        if args.global_merge_sim_threshold is not None
        else _cfg(loaded, "clustering.global_merge_similarity_threshold", None)
    )
    raw_out = Path(args.raw_out or _cfg(loaded, "output.raw_out", str(REPO_ROOT / "data" / "Processed" / "user_turns_raw.jsonl")))
    dedup_out = Path(args.dedup_out or _cfg(loaded, "output.dedup_out", str(REPO_ROOT / "data" / "Processed" / "user_turns_dedup.jsonl")))
    memory_out = Path(args.memory_out or _cfg(loaded, "output.memory_out", str(REPO_ROOT / "data" / "Processed" / "memory_chat.jsonl")))

    output = build_processed_turns(
        input_path,
        config=ProcessorConfig(
            session_id=session_id or None,
            segmentation_mode=segmentation_mode,
            fixed_sim_threshold=float(fixed_sim_threshold),
            novelty_threshold=float(novelty_threshold),
            min_segment_len=int(min_segment_len),
            cross_session_merge=bool(cross_session_merge),
            global_merge_similarity_threshold=(
                float(global_merge_sim_threshold)
                if global_merge_sim_threshold is not None
                else None
            ),
        ),
    )

    write_jsonl(raw_out, output.raw_rows)
    write_jsonl(dedup_out, output.dedup_rows)
    write_jsonl(memory_out, output.memory_rows)

    excluded = sum(1 for row in output.raw_rows if row.get("is_excluded"))
    print(f"raw_rows={len(output.raw_rows)} excluded={excluded} => {raw_out}")
    print(f"dedup_rows={len(output.dedup_rows)} => {dedup_out}")
    print(f"memory_rows={len(output.memory_rows)} => {memory_out}")
    print("query/eval build is separated: run scripts/memory_processer/build_chat_queries.py")


if __name__ == "__main__":
    main()
