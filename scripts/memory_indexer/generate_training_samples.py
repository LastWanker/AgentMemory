"""Generate listwise-ready followup samples and processed eval datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MEMORY_EVAL_DIR = DATA_DIR / "Memory_Eval"
PROCESSED_DIR = DATA_DIR / "Processed"


def load_logs(path: Path) -> Dict[str, List[Dict[str, object]]]:
    sessions: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        sessions[str(payload.get("session_id", "na"))].append(payload)
    for turns in sessions.values():
        turns.sort(key=lambda row: int(row.get("turn_index", 0)))
    return sessions


def normalize_cites(row: Dict[str, object]) -> List[str]:
    values = row.get("cited_mem_ids", [])
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for mid in values:
        text = str(mid).strip()
        if text:
            out.append(text)
    return out


def build_samples_from_logs(sessions: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    """No hard dependency on user_feedback: use cited_mem_ids continuity."""

    samples: List[Dict[str, object]] = []
    for session_id, turns in sessions.items():
        for i in range(1, len(turns)):
            prev_turn = turns[i - 1]
            curr_turn = turns[i]
            curr_cites = set(normalize_cites(curr_turn))
            if not curr_cites:
                continue
            prev_cites = set(normalize_cites(prev_turn))
            positives = sorted(prev_cites & curr_cites)
            if not positives:
                positives = sorted(curr_cites)
            if not positives:
                continue
            query_text = str(curr_turn.get("query_text", "")).strip()
            if not query_text:
                continue
            samples.append(
                {
                    "query_id": f"{session_id}-{curr_turn.get('turn_index', i)}",
                    "query_text": query_text,
                    "positives": positives,
                    "meta": {
                        "source": "logs",
                        "session_id": session_id,
                        "turn_index": curr_turn.get("turn_index", i),
                    },
                }
            )
    return samples


def resolve_memory_path(dataset: str) -> Path:
    processed = PROCESSED_DIR / f"memory_{dataset}.jsonl"
    legacy = MEMORY_EVAL_DIR / f"memory_{dataset}.jsonl"
    if processed.exists():
        return processed
    if legacy.exists():
        return legacy
    fallback = DATA_DIR / f"memory_{dataset}.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"memory dataset not found for '{dataset}'")


def resolve_eval_input_path(dataset: str) -> Path:
    processed = PROCESSED_DIR / f"eval_{dataset}.jsonl"
    legacy = MEMORY_EVAL_DIR / f"eval_{dataset}.jsonl"
    if processed.exists():
        return processed
    if legacy.exists():
        return legacy
    fallback = DATA_DIR / f"eval_{dataset}.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"eval dataset not found for '{dataset}'")


def resolve_eval_output_path(dataset: str, explicit_out: Optional[str]) -> Path:
    if explicit_out:
        return Path(explicit_out)
    return PROCESSED_DIR / f"eval_{dataset}.jsonl"


def ensure_followup_memory(dataset: str, source_dataset: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = PROCESSED_DIR / f"memory_{dataset}.jsonl"
    if target.exists():
        return target
    src = resolve_memory_path(source_dataset)
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def normalize_eval_positives(payload: Dict[str, object]) -> List[str]:
    positives = payload.get("positives")
    if isinstance(positives, list):
        return [str(x) for x in positives if str(x).strip()]
    expected = payload.get("expected_mem_ids")
    if isinstance(expected, list):
        return [str(x) for x in expected if str(x).strip()]
    return []


def build_samples_from_eval_dataset(dataset: str) -> List[Dict[str, object]]:
    input_path = resolve_eval_input_path(dataset)
    rows = []
    for idx, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        query_text = str(payload.get("query_text", "")).strip()
        positives = normalize_eval_positives(payload)
        if not query_text or not positives:
            continue
        rows.append(
            {
                "query_id": str(payload.get("query_id") or f"synth-{idx}"),
                "query_text": query_text,
                "positives": positives,
                "meta": {
                    "source": "eval_synth",
                    "from_eval_dataset": dataset,
                    "row_index": idx,
                },
            }
        )
    return rows


def build_eval_rows(
    samples: List[Dict[str, object]],
    *,
    dataset: str,
    top_n: int,
    encoder_backend: str,
    candidate_mode: str,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in samples:
        query_text = str(row.get("query_text", "")).strip()
        positives = [str(mid) for mid in (row.get("positives") or []) if str(mid).strip()]
        if not query_text or not positives:
            continue

        enriched.append(
            {
                "query_id": row.get("query_id"),
                "query_text": query_text,
                "positives": positives,
                "meta": {
                    **(row.get("meta") if isinstance(row.get("meta"), dict) else {}),
                    "source": "followup",
                    "dataset": dataset,
                    # 候选池统一在评测/训练时由粗召回 top_n 动态生成，不再落盘到 query。
                    "candidate_mode": candidate_mode,
                    "candidate_top_n": top_n,
                    "encoder_backend": encoder_backend,
                },
            }
        )
    return enriched


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training/listwise samples.")
    parser.add_argument("--log-path", help="Input interaction log jsonl.")
    parser.add_argument("--output-path", help="Legacy mode output path for raw samples.")
    parser.add_argument("--export-eval", action="store_true", help="Export processed eval dataset.")
    parser.add_argument("--out", help="Output path for exported eval dataset.")
    parser.add_argument("--dataset", default="followup", help="Target dataset name for output.")
    parser.add_argument(
        "--source-memory-dataset",
        default="normal",
        help="Fallback source memory dataset for memory_followup bootstrap.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Metadata only; runtime retrieval budget.")
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="coarse")
    parser.add_argument("--encoder-backend", choices=("simple", "hf"), default="simple")
    parser.add_argument(
        "--from-eval-dataset",
        help="Synthesize followup samples from an existing eval dataset (e.g. normal).",
    )
    args = parser.parse_args()

    if args.from_eval_dataset:
        samples = build_samples_from_eval_dataset(args.from_eval_dataset)
    else:
        if not args.log_path:
            raise ValueError("Either --log-path or --from-eval-dataset is required.")
        sessions = load_logs(Path(args.log_path))
        samples = build_samples_from_logs(sessions)

    if not args.export_eval:
        if not args.output_path:
            raise ValueError("When --export-eval is off, --output-path is required.")
        out = Path(args.output_path)
        write_jsonl(out, samples)
        print(f"generated_samples={len(samples)} => {out}")
        return

    memory_followup = ensure_followup_memory(args.dataset, args.source_memory_dataset)
    enriched = build_eval_rows(
        samples,
        dataset=args.dataset,
        top_n=args.top_n,
        encoder_backend=args.encoder_backend,
        candidate_mode=args.candidate_mode,
    )
    out = resolve_eval_output_path(args.dataset, args.out)
    write_jsonl(out, enriched)
    print(f"generated_followup_eval={len(enriched)} => {out}")
    print(f"memory_dataset_ready => {memory_followup}")


if __name__ == "__main__":
    main()

