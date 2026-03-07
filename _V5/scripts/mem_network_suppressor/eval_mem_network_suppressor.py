from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypeVar


def _resolve_v5_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src").exists() and (parent / "chat" / "src").exists():
            return parent
    return here.parents[1]


V5_ROOT = _resolve_v5_root()
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))
if str(V5_ROOT / "chat" / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "chat" / "src"))

from agentmemory_v3.suppressor.data_utils import load_feedback_rows
from chat_app.config import load_config
from chat_app.models import MemoryRef
from chat_app.retriever_adapter import RetrieverAdapter
from chat_app.suppressor_adapter import SuppressorAdapter


T = TypeVar("T")


def _ensure_non_empty(values: list[T], fallback: T) -> list[T]:
    return values if values else [fallback]


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for chunk in str(value or "").split(","):
        text = chunk.strip()
        if text:
            out.append(float(text))
    return out


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for chunk in str(value or "").split(","):
        text = chunk.strip()
        if text:
            out.append(int(text))
    return out


def _unique_canonical_names(path: Path, sample_size: int, seed: int) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = str(row.get("canonical_name") or "").strip()
        if text:
            names.append(text)
    unique = list(dict.fromkeys(names))
    random.Random(int(seed)).shuffle(unique)
    return unique[: max(1, int(sample_size))]


def _clone_refs(refs: list[MemoryRef]) -> list[MemoryRef]:
    return [MemoryRef.model_validate(ref.model_dump()) for ref in refs]


def _rank_1based(values: list[str], memory_id: str) -> int | None:
    try:
        return int(values.index(memory_id)) + 1
    except ValueError:
        return None


def _precompute_baseline(
    *,
    query_texts: list[str],
    pool_top_k: int,
) -> dict[str, dict]:
    cfg = load_config()
    cfg.suppressor_enabled = False
    retriever = RetrieverAdapter(cfg)
    baseline: dict[str, dict] = {}
    for idx, query in enumerate(query_texts, start=1):
        bundle = retriever.retrieve_bundle(query, pool_top_k, memory_preference_enabled=False)
        baseline[query] = {
            "coarse": _clone_refs(bundle.coarse_refs),
            "association": _clone_refs(bundle.association_refs),
        }
        if idx % 200 == 0:
            print("[baseline_progress]", idx, "/", len(query_texts))
    return baseline


def _apply_combo_to_query(query: str, base_row: dict, adapter: SuppressorAdapter, *, top_k: int) -> dict:
    coarse_before_refs = _clone_refs(base_row["coarse"])
    association_before_refs = _clone_refs(base_row["association"])
    coarse_before = [item.memory_id for item in coarse_before_refs]
    association_before = [item.memory_id for item in association_before_refs]
    coarse_after_refs, coarse_trace = adapter.apply(query, "coarse", coarse_before_refs, enabled_override=True)
    association_after_refs, association_trace = adapter.apply(query, "association", association_before_refs, enabled_override=True)
    coarse_after = [item.memory_id for item in coarse_after_refs]
    association_after = [item.memory_id for item in association_after_refs]
    k = max(1, int(top_k))
    return {
        "coarse_off_full": coarse_before,
        "coarse_on_full": coarse_after,
        "association_off_full": association_before,
        "association_on_full": association_after,
        "coarse_off_topk": coarse_before[:k],
        "coarse_on_topk": coarse_after[:k],
        "association_off_topk": association_before[:k],
        "association_on_topk": association_after[:k],
        "coarse_trace": coarse_trace,
        "association_trace": association_trace,
    }


def _evaluate_combo(
    *,
    combo_cfg,
    baseline: dict[str, dict],
    feedback_rows: list[dict],
    feedback_memory_id_set: set[str],
    random_queries: list[str],
    top_k: int,
) -> dict:
    adapter = SuppressorAdapter(combo_cfg)
    query_result_cache: dict[str, dict] = {}

    def query_result(query: str) -> dict:
        cached = query_result_cache.get(query)
        if cached is not None:
            return cached
        row = _apply_combo_to_query(query, baseline[query], adapter, top_k=top_k)
        query_result_cache[query] = row
        return row

    feedback_pair_hit_off = 0
    feedback_pair_removed = 0
    for row in feedback_rows:
        query = str(row.get("q_text") or "")
        memory_id = str(row.get("m_id") or "")
        lane = str(row.get("lane") or "").strip().lower()
        if query not in baseline:
            continue
        qr = query_result(query)
        off_ids = qr["association_off_topk"] if lane == "association" else qr["coarse_off_topk"]
        on_ids = qr["association_on_topk"] if lane == "association" else qr["coarse_on_topk"]
        if memory_id in off_ids:
            feedback_pair_hit_off += 1
            if memory_id not in on_ids:
                feedback_pair_removed += 1

    random_top1_changed = 0
    random_top5_set_changed = 0
    random_feedback_hit_off_count = 0
    random_feedback_hit_on_count = 0
    random_feedback_removed_count = 0
    random_feedback_hit_queries_off = 0
    random_feedback_hit_queries_on = 0
    random_feedback_hit_unique_off: set[str] = set()
    random_feedback_hit_unique_on: set[str] = set()
    random_feedback_removed_unique: set[str] = set()
    for query in random_queries:
        qr = query_result(query)
        c1 = (qr["coarse_off_topk"][:1] != qr["coarse_on_topk"][:1])
        a1 = (qr["association_off_topk"][:1] != qr["association_on_topk"][:1])
        cs = (set(qr["coarse_off_topk"]) != set(qr["coarse_on_topk"]))
        aset = (set(qr["association_off_topk"]) != set(qr["association_on_topk"]))
        if c1 or a1:
            random_top1_changed += 1
        if cs or aset:
            random_top5_set_changed += 1

        lane_off_sets = [
            set(qr["coarse_off_topk"]) & feedback_memory_id_set,
            set(qr["association_off_topk"]) & feedback_memory_id_set,
        ]
        lane_on_sets = [
            set(qr["coarse_on_topk"]) & feedback_memory_id_set,
            set(qr["association_on_topk"]) & feedback_memory_id_set,
        ]
        off_any = False
        on_any = False
        for off_set, on_set in zip(lane_off_sets, lane_on_sets):
            if off_set:
                off_any = True
            if on_set:
                on_any = True
            random_feedback_hit_off_count += int(len(off_set))
            random_feedback_hit_on_count += int(len(on_set))
            random_feedback_hit_unique_off.update(off_set)
            random_feedback_hit_unique_on.update(on_set)
            removed_set = off_set - on_set
            random_feedback_removed_count += int(len(removed_set))
            random_feedback_removed_unique.update(removed_set)
        if off_any:
            random_feedback_hit_queries_off += 1
        if on_any:
            random_feedback_hit_queries_on += 1

    random_count = max(1, len(random_queries))
    feedback_unique_memory_count = max(1, len(feedback_memory_id_set))
    feedback_rate = float(feedback_pair_removed) / float(max(1, feedback_pair_hit_off))
    top1_rate = float(random_top1_changed) / float(random_count)
    top5_rate = float(random_top5_set_changed) / float(random_count)
    score = 1.5 * feedback_rate - 1.5 * top1_rate - 1.2 * top5_rate
    return {
        "threshold": float(combo_cfg.suppressor_memnet_score_threshold),
        "max_drop_per_lane": int(combo_cfg.suppressor_memnet_max_drop_per_lane),
        "keep_top_per_lane": int(combo_cfg.suppressor_memnet_keep_top_per_lane),
        "feedback_pair_hit_off_count": int(feedback_pair_hit_off),
        "feedback_pair_removed_count": int(feedback_pair_removed),
        "feedback_pair_removed_rate": feedback_rate,
        "random_query_count": int(random_count),
        "random_top1_changed_rate": top1_rate,
        "random_top5_set_changed_rate": top5_rate,
        "random_feedback_hit_queries_off_count": int(random_feedback_hit_queries_off),
        "random_feedback_hit_queries_on_count": int(random_feedback_hit_queries_on),
        "random_feedback_hit_queries_off_rate": float(random_feedback_hit_queries_off) / float(random_count),
        "random_feedback_hit_queries_on_rate": float(random_feedback_hit_queries_on) / float(random_count),
        "random_feedback_memory_hit_off_count": int(random_feedback_hit_off_count),
        "random_feedback_memory_hit_on_count": int(random_feedback_hit_on_count),
        "random_feedback_memory_removed_count": int(random_feedback_removed_count),
        "random_feedback_memory_removed_when_hit_rate": (
            float(random_feedback_removed_count) / float(max(1, random_feedback_hit_off_count))
        ),
        "random_feedback_memory_hit_unique_off_count": int(len(random_feedback_hit_unique_off)),
        "random_feedback_memory_hit_unique_on_count": int(len(random_feedback_hit_unique_on)),
        "random_feedback_memory_removed_unique_count": int(len(random_feedback_removed_unique)),
        "random_feedback_memory_hit_unique_ratio_off": (
            float(len(random_feedback_hit_unique_off)) / float(feedback_unique_memory_count)
        ),
        "random_feedback_memory_hit_unique_ratio_on": (
            float(len(random_feedback_hit_unique_on)) / float(feedback_unique_memory_count)
        ),
        "random_feedback_memory_removed_unique_ratio": (
            float(len(random_feedback_removed_unique)) / float(feedback_unique_memory_count)
        ),
        # Compatibility aliases for older reports.
        "random_dual_top1_changed_rate": top1_rate,
        "random_dual_top5_set_changed_rate": top5_rate,
        "score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mem-network suppressor by two cohorts.")
    parser.add_argument("--feedback-jsonl", default="_V5/chat/data/feedback/feedback_events.jsonl")
    parser.add_argument("--concepts-jsonl", default="data/V5/association/concepts.jsonl")
    parser.add_argument("--artifact-dir", default="data/V5/mem_network_suppressor/current")
    parser.add_argument("--random-sample-size", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-extra", type=int, default=8)
    parser.add_argument("--thresholds", default="0.35,0.45,0.55")
    parser.add_argument("--max-drops", default="0")
    parser.add_argument("--keep-tops", default="0")
    parser.add_argument("--output-json", default="data/V5/mem_network_suppressor/eval_two_cohort.json")
    args = parser.parse_args()

    feedback_rows = load_feedback_rows(args.feedback_jsonl)
    if not feedback_rows:
        raise RuntimeError("no feedback rows loaded")
    feedback_memory_id_set = {
        str(row.get("m_id") or "").strip()
        for row in feedback_rows
        if str(row.get("m_id") or "").strip()
    }
    feedback_queries = list(dict.fromkeys(str(row.get("q_text") or "") for row in feedback_rows if str(row.get("q_text") or "")))
    random_queries = _unique_canonical_names(Path(args.concepts_jsonl), int(args.random_sample_size), int(args.random_seed))
    all_queries = list(dict.fromkeys(feedback_queries + random_queries))
    pool_top_k = max(1, int(args.top_k) + max(0, int(args.pool_extra)))
    baseline = _precompute_baseline(query_texts=all_queries, pool_top_k=pool_top_k)

    thresholds = _ensure_non_empty(_parse_float_list(args.thresholds), 0.45)
    _ = _ensure_non_empty(_parse_int_list(args.max_drops), 0)
    _ = _ensure_non_empty(_parse_int_list(args.keep_tops), 0)

    base_cfg = load_config()
    results: list[dict] = []
    idx = 0
    total = len(thresholds)
    for threshold in thresholds:
        idx += 1
        combo_cfg = replace(
            base_cfg,
            suppressor_enabled=True,
            suppressor_backend="mem_network",
            suppressor_memnet_artifact_dir=Path(args.artifact_dir),
            suppressor_memnet_score_threshold=float(threshold),
            suppressor_memnet_max_drop_per_lane=0,
            suppressor_memnet_keep_top_per_lane=0,
            suppressor_extra_candidates=int(args.pool_extra),
        )
        row = _evaluate_combo(
            combo_cfg=combo_cfg,
            baseline=baseline,
            feedback_rows=feedback_rows,
            feedback_memory_id_set=feedback_memory_id_set,
            random_queries=random_queries,
            top_k=max(1, int(args.top_k)),
        )
        results.append(row)
        print(
            "[combo]",
            f"{idx}/{total}",
            {
                "thr": threshold,
                "max_drop": 0,
                "keep_top": 0,
                "feedback_removed": round(float(row["feedback_pair_removed_rate"]), 4),
                "rand_top1": round(float(row["random_top1_changed_rate"]), 4),
                "rand_set": round(float(row["random_top5_set_changed_rate"]), 4),
                "rand_fb_hit_u": round(float(row["random_feedback_memory_hit_unique_ratio_off"]), 4),
                "rand_fb_removed": round(float(row["random_feedback_memory_removed_when_hit_rate"]), 4),
                "score": round(float(row["score"]), 4),
            },
        )

    results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    best = results[0] if results else {}
    payload = {
        "artifact_dir": str(args.artifact_dir),
        "random_sample_size": int(len(random_queries)),
        "feedback_pair_count": int(len(feedback_rows)),
        "feedback_unique_memory_count": int(len(feedback_memory_id_set)),
        "combo_count": int(len(results)),
        "grid": {
            "thresholds": thresholds,
            "max_drops": [0],
            "keep_tops": [0],
        },
        "best": best,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v5][memnet][eval]", {"combo_count": len(results), "output_json": str(output_path), "best": best})


if __name__ == "__main__":
    main()
