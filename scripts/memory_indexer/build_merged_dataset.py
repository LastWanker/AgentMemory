"""Merge two processed datasets into one train/eval dataset namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "Processed"


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_memory_rows(
    rows_a: List[Dict[str, object]],
    rows_b: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], int]:
    merged: List[Dict[str, object]] = []
    seen_ids = set()
    duplicate_ids = 0

    for row in rows_a + rows_b:
        mem_id = str(row.get("mem_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not mem_id or not text:
            continue
        if mem_id in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(mem_id)
        merged.append(row)
    return merged, duplicate_ids


def ensure_unique_query_id(base_id: str, seen_ids: set[str], fallback_prefix: str, seq: int) -> Tuple[str, int]:
    candidate = base_id or f"{fallback_prefix}-{seq:06d}"
    while candidate in seen_ids:
        seq += 1
        candidate = f"{fallback_prefix}-{seq:06d}"
    seen_ids.add(candidate)
    return candidate, seq


def merge_eval_rows(
    sources: List[Tuple[str, List[Dict[str, object]]]],
) -> Tuple[List[Dict[str, object]], int]:
    merged: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    seq = 0
    renamed = 0

    for source_name, rows in sources:
        for row in rows:
            query_text = str(row.get("query_text", "")).strip()
            if not query_text:
                continue
            seq += 1
            orig_id = str(row.get("query_id", "")).strip()
            query_id, seq = ensure_unique_query_id(orig_id, seen_ids, "mixq", seq)
            if query_id != orig_id:
                renamed += 1
            new_row = dict(row)
            new_row["query_id"] = query_id

            meta = new_row.get("meta")
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
            meta_dict.setdefault("merged_source", source_name)
            new_row["meta"] = meta_dict
            merged.append(new_row)

    return merged, renamed


def extract_positives(row: Dict[str, object]) -> List[str]:
    if isinstance(row.get("positives"), list):
        return [str(x).strip() for x in row["positives"] if str(x).strip()]
    if isinstance(row.get("expected_mem_ids"), list):
        return [str(x).strip() for x in row["expected_mem_ids"] if str(x).strip()]
    return []


def validate_eval_links(eval_rows: List[Dict[str, object]], memory_rows: List[Dict[str, object]]) -> int:
    mem_ids = {str(row.get("mem_id", "")).strip() for row in memory_rows}
    missing = 0
    for row in eval_rows:
        positives = extract_positives(row)
        if positives and any(pos not in mem_ids for pos in positives):
            missing += 1
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge followup/chat processed datasets.")
    parser.add_argument("--dataset", default="followup_plus_chat", help="Output dataset suffix.")
    parser.add_argument("--memory-a", default=str(PROCESSED_DIR / "memory_followup.jsonl"))
    parser.add_argument("--eval-a", default=str(PROCESSED_DIR / "eval_followup.jsonl"))
    parser.add_argument("--memory-b", default=str(PROCESSED_DIR / "memory_chat.jsonl"))
    parser.add_argument("--eval-b", default=str(PROCESSED_DIR / "eval_chat_followup.jsonl"))
    parser.add_argument("--eval-c", default=str(PROCESSED_DIR / "eval_chat_supplemental.api.jsonl"))
    parser.add_argument("--source-a-name", default="followup")
    parser.add_argument("--source-b-name", default="chat_identity")
    parser.add_argument("--source-c-name", default="chat_supplemental")
    parser.add_argument(
        "--group-dir",
        default=None,
        help="Optional grouped output folder (default: data/Processed/groups/<dataset>).",
    )
    parser.add_argument("--no-group-copy", action="store_true")
    args = parser.parse_args()

    memory_a = Path(args.memory_a)
    eval_a = Path(args.eval_a)
    memory_b = Path(args.memory_b)
    eval_b = Path(args.eval_b)
    eval_c = Path(args.eval_c) if str(args.eval_c).strip() else None
    if not memory_a.exists() or not eval_a.exists() or not memory_b.exists() or not eval_b.exists():
        raise FileNotFoundError(
            "missing input files. "
            f"memory_a={memory_a.exists()} eval_a={eval_a.exists()} "
            f"memory_b={memory_b.exists()} eval_b={eval_b.exists()}"
        )
    if eval_c and not eval_c.exists():
        raise FileNotFoundError(f"missing input file: eval_c={eval_c}")

    memory_rows_a = read_jsonl(memory_a)
    memory_rows_b = read_jsonl(memory_b)
    eval_rows_a = read_jsonl(eval_a)
    eval_rows_b = read_jsonl(eval_b)

    merged_memory, duplicate_mem_ids = merge_memory_rows(memory_rows_a, memory_rows_b)
    eval_sources: List[Tuple[str, List[Dict[str, object]]]] = [
        (args.source_a_name, eval_rows_a),
        (args.source_b_name, eval_rows_b),
    ]
    eval_rows_c: List[Dict[str, object]] = []
    if eval_c:
        eval_rows_c = read_jsonl(eval_c)
        eval_sources.append((args.source_c_name, eval_rows_c))
    merged_eval, renamed_query_ids = merge_eval_rows(eval_sources)
    missing_positive_rows = validate_eval_links(merged_eval, merged_memory)

    out_memory = PROCESSED_DIR / f"memory_{args.dataset}.jsonl"
    out_eval = PROCESSED_DIR / f"eval_{args.dataset}.jsonl"
    write_jsonl(out_memory, merged_memory)
    write_jsonl(out_eval, merged_eval)

    group_dir = Path(args.group_dir) if args.group_dir else (PROCESSED_DIR / "groups" / args.dataset)
    if not args.no_group_copy:
        write_jsonl(group_dir / "memory.jsonl", merged_memory)
        write_jsonl(group_dir / "eval.jsonl", merged_eval)

    print(f"memory_a_rows={len(memory_rows_a)} <= {memory_a}")
    print(f"memory_b_rows={len(memory_rows_b)} <= {memory_b}")
    print(f"eval_a_rows={len(eval_rows_a)} <= {eval_a}")
    print(f"eval_b_rows={len(eval_rows_b)} <= {eval_b}")
    if eval_c:
        print(f"eval_c_rows={len(eval_rows_c)} <= {eval_c}")
    print(f"merged_memory_rows={len(merged_memory)} => {out_memory}")
    print(f"merged_eval_rows={len(merged_eval)} => {out_eval}")
    print(
        f"stats: duplicate_mem_ids={duplicate_mem_ids} "
        f"renamed_query_ids={renamed_query_ids} missing_positive_rows={missing_positive_rows}"
    )
    if not args.no_group_copy:
        print(f"grouped_memory={group_dir / 'memory.jsonl'}")
        print(f"grouped_eval={group_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()

