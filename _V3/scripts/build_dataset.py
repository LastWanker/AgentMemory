from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import resolve_path
from agentmemory_v3.utils.io import read_jsonl, write_jsonl


def _safe_turn_order(value: object) -> tuple[int, str]:
    text = str(value or "")
    try:
        return 0, str(int(text))
    except ValueError:
        return 1, text


def build_dataset(memory_in: Path, eval_in: Path, out_root: Path) -> dict:
    memory_rows = list(read_jsonl(memory_in))
    eval_rows = list(read_jsonl(eval_in, encoding="utf-8-sig"))

    grouped_sessions: dict[str, list[dict]] = defaultdict(list)
    for row in memory_rows:
        grouped_sessions[str(row.get("session_id") or "")].append(row)

    prev_next_map: dict[str, tuple[str, str]] = {}
    for _, rows in grouped_sessions.items():
        rows.sort(key=lambda row: _safe_turn_order(row.get("turn_id")))
        for idx, row in enumerate(rows):
            mem_id = str(row.get("mem_id") or "")
            prev_text = str(rows[idx - 1].get("text") or "") if idx > 0 else ""
            next_text = str(rows[idx + 1].get("text") or "") if idx + 1 < len(rows) else ""
            prev_next_map[mem_id] = (prev_text, next_text)

    memory_out = []
    mem_to_cluster: dict[str, str] = {}
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in memory_rows:
        memory_id = str(row.get("mem_id") or "")
        cluster_id = str(row.get("cluster_id") or memory_id)
        prev_raw, next_raw = prev_next_map.get(memory_id, ("", ""))
        record = {
            "memory_id": memory_id,
            "cluster_id": cluster_id,
            "raw_text": str(row.get("text") or ""),
            "session_id": str(row.get("session_id") or ""),
            "turn_id": str(row.get("turn_id") or ""),
            "source": str(row.get("source") or ""),
            "tags": list(row.get("tags") or []),
            "prev_raw": prev_raw,
            "next_raw": next_raw,
        }
        memory_out.append(record)
        mem_to_cluster[memory_id] = cluster_id
        clusters[cluster_id].append(memory_id)

    query_out = []
    for row in eval_rows:
        positives = [str(item) for item in (row.get("positives") or row.get("expected_mem_ids") or []) if str(item).strip()]
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        cluster_candidates = {mem_to_cluster.get(mem_id, mem_id) for mem_id in positives}
        cluster_id = next(iter(cluster_candidates)) if len(cluster_candidates) == 1 else ""
        query_out.append(
            {
                "query_id": str(row.get("query_id") or ""),
                "text": str(row.get("query_text") or ""),
                "positives": positives,
                "cluster_id": cluster_id,
                "source": str(meta.get("source") or row.get("source") or ""),
                "channel": str(meta.get("channel") or row.get("channel") or ""),
                "meta": meta,
            }
        )

    cluster_out = [{"cluster_id": cluster_id, "memory_ids": mem_ids, "size": len(mem_ids)} for cluster_id, mem_ids in clusters.items()]

    processed_dir = out_root / "processed"
    write_jsonl(processed_dir / "memory.jsonl", memory_out)
    write_jsonl(processed_dir / "query.jsonl", query_out)
    write_jsonl(processed_dir / "cluster.jsonl", cluster_out)
    return {
        "memory_count": len(memory_out),
        "query_count": len(query_out),
        "cluster_count": len(cluster_out),
        "processed_dir": str(processed_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 dataset.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--out-root", default="data/V3")
    args = parser.parse_args()
    result = build_dataset(
        resolve_path(args.memory_in),
        resolve_path(args.eval_in),
        resolve_path(args.out_root),
    )
    print(
        f"[v3] dataset memory={result['memory_count']} query={result['query_count']} "
        f"cluster={result['cluster_count']} -> {result['processed_dir']}"
    )


if __name__ == "__main__":
    main()
