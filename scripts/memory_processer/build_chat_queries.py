"""Build chat eval datasets from processed memory rows.

Base behavior:
- identity queries only: query_text == memory text

Optional behavior:
- append supplemental queries from an external JSONL file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chat_memory_processor import (  # noqa: E402
    build_query_samples,
    to_eval_rows,
    to_followup_rows,
    write_jsonl,
)


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


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval/followup query datasets from memory rows.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "chat_memory.yaml"),
        help="Optional JSON/YAML config file.",
    )
    parser.add_argument("--memory-in", default=None, help="Input memory jsonl.")
    parser.add_argument("--eval-out", default=None, help="Output eval jsonl.")
    parser.add_argument("--followup-out", default=None, help="Output followup jsonl.")
    parser.add_argument(
        "--supplemental-queries",
        default=None,
        help="Optional external JSONL for appended supplemental queries.",
    )
    args = parser.parse_args()

    loaded = _load_config(Path(args.config))
    memory_in = Path(
        args.memory_in or _cfg(loaded, "output.memory_out", str(REPO_ROOT / "data" / "Processed" / "memory_chat.jsonl"))
    )
    eval_out = Path(args.eval_out or _cfg(loaded, "output.eval_out", str(REPO_ROOT / "data" / "Processed" / "eval_chat.jsonl")))
    followup_out = Path(
        args.followup_out
        or _cfg(loaded, "output.followup_out", str(REPO_ROOT / "data" / "Processed" / "eval_chat_followup.jsonl"))
    )
    supplemental_path_raw = (
        args.supplemental_queries
        if args.supplemental_queries is not None
        else _cfg(loaded, "query.supplemental_queries", None)
    )
    supplemental_path = Path(supplemental_path_raw) if supplemental_path_raw else None

    memories = _read_jsonl(memory_in)
    samples = build_query_samples(memories, supplemental_path=supplemental_path)
    followup_rows = to_followup_rows(samples)
    eval_rows = to_eval_rows(samples)

    write_jsonl(followup_out, followup_rows)
    write_jsonl(eval_out, eval_rows)

    print(f"memory_rows={len(memories)} <= {memory_in}")
    print(f"query_rows={len(samples)} => {followup_out}")
    print(f"eval_rows={len(eval_rows)} => {eval_out}")
    if supplemental_path:
        print(f"supplemental_queries={supplemental_path}")
    else:
        print("supplemental_queries=<none>")


if __name__ == "__main__":
    main()
