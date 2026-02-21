"""Generate listwise-ready followup samples and processed eval datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_indexer import (  # noqa: E402
    SimpleHashEncoder,
    Vectorizer,
    build_memory_index,
    build_memory_items,
    retrieve_top_k,
)

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
            hard_negatives = sorted(prev_cites - set(positives))
            query_text = str(curr_turn.get("query_text", "")).strip()
            if not query_text:
                continue
            samples.append(
                {
                    "query_id": f"{session_id}-{curr_turn.get('turn_index', i)}",
                    "query_text": query_text,
                    "positives": positives,
                    "hard_negatives": hard_negatives,
                    "candidates": sorted(prev_cites | curr_cites),
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
                "hard_negatives": [],
                "candidates": [],
                "meta": {
                    "source": "eval_synth",
                    "from_eval_dataset": dataset,
                    "row_index": idx,
                },
            }
        )
    return rows


def configure_hf_runtime(local_only: bool, offline: bool) -> None:
    os.environ["HF_LOCAL_FILES_ONLY"] = "1" if local_only else "0"
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"


def build_candidates_for_samples(
    samples: List[Dict[str, object]],
    *,
    dataset: str,
    top_n: int,
    encoder_backend: str,
    candidate_mode: str,
) -> List[Dict[str, object]]:
    memory_path = resolve_memory_path(dataset)
    payloads = [
        json.loads(line)
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    memory_items = build_memory_items(
        payloads,
        source_default="manual",
        chunk_strategy="sentence_window",
        max_sentences=3,
        tags_default=["general"],
    )

    if encoder_backend == "simple":
        sentence_encoder = SimpleHashEncoder(dims=64)
        token_encoder = sentence_encoder
    else:
        from src.memory_indexer.encoder.e5_token import E5TokenEncoder  # noqa: E402
        from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder  # noqa: E402

        configure_hf_runtime(local_only=True, offline=True)
        sentence_encoder = HFSentenceEncoder(
            model_name="intfloat/multilingual-e5-small",
            tokenizer="jieba",
        )
        token_encoder = E5TokenEncoder(model_name="intfloat/multilingual-e5-small")

    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    store, index, lexical_index = build_memory_index(
        memory_items,
        sentence_encoder,
        vectorizer,
        token_encoder=token_encoder,
        cache_path=None,
        return_lexical=True,
    )
    all_mem_ids = list(store.items.keys())

    enriched: List[Dict[str, object]] = []
    for row in samples:
        query_text = str(row.get("query_text", "")).strip()
        positives = [str(mid) for mid in (row.get("positives") or []) if str(mid).strip()]
        if not query_text or not positives:
            continue

        results = retrieve_top_k(
            query_text,
            sentence_encoder,
            vectorizer,
            store,
            index,
            lexical_index=lexical_index,
            top_n=top_n,
            top_k=top_n,
            candidate_mode=candidate_mode,
            token_encoder=token_encoder,
        )
        candidates = [item.mem_id for item in results]
        for pos in positives:
            if pos not in candidates:
                candidates.insert(0, pos)

        seen = set()
        deduped_candidates: List[str] = []
        for mem_id in candidates:
            if mem_id in seen:
                continue
            seen.add(mem_id)
            deduped_candidates.append(mem_id)
        target_size = min(top_n, len(all_mem_ids))
        if len(deduped_candidates) < target_size:
            for mem_id in all_mem_ids:
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                deduped_candidates.append(mem_id)
                if len(deduped_candidates) >= target_size:
                    break

        hard_negatives = [
            str(mid)
            for mid in (row.get("hard_negatives") or [])
            if str(mid).strip() and str(mid) not in set(positives)
        ]
        if not hard_negatives:
            hard_negatives = [mid for mid in deduped_candidates if mid not in set(positives)][:10]

        enriched.append(
            {
                "query_id": row.get("query_id"),
                "query_text": query_text,
                "positives": positives,
                "candidates": deduped_candidates,
                "hard_negatives": hard_negatives,
                "meta": {
                    **(row.get("meta") if isinstance(row.get("meta"), dict) else {}),
                    "source": "followup",
                    "dataset": dataset,
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
    parser.add_argument("--top-n", type=int, default=20, help="Target minimum candidates per sample.")
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="union")
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
    enriched = build_candidates_for_samples(
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
