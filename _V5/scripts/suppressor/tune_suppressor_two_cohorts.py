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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(p))) * float(len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = float(pos - float(lo))
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


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


def _apply_combo_to_query(
    query: str,
    base_row: dict,
    adapter: SuppressorAdapter,
    *,
    top_k: int,
) -> dict:
    coarse_before_refs = _clone_refs(base_row["coarse"])
    association_before_refs = _clone_refs(base_row["association"])
    coarse_before = [item.memory_id for item in coarse_before_refs]
    association_before = [item.memory_id for item in association_before_refs]
    coarse_after_refs, coarse_trace = adapter.apply(query, "coarse", coarse_before_refs, enabled_override=True)
    association_after_refs, association_trace = adapter.apply(
        query,
        "association",
        association_before_refs,
        enabled_override=True,
    )
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
    feedback_queries: list[str],
    random_queries: list[str],
    top_k: int,
) -> dict:
    adapter = SuppressorAdapter(combo_cfg)
    query_result_cache: dict[str, dict] = {}

    def query_result(query: str) -> dict:
        cached = query_result_cache.get(query)
        if cached is not None:
            return cached
        row = _apply_combo_to_query(
            query,
            baseline[query],
            adapter,
            top_k=top_k,
        )
        query_result_cache[query] = row
        return row

    feedback_pair_count = len(feedback_rows)
    feedback_pair_hit_off_count = 0
    feedback_pair_removed_count = 0
    feedback_pair_top1_hit_off_count = 0
    feedback_pair_top1_removed_count = 0
    feedback_rank_shift_all: list[float] = []
    feedback_rank_shift_topk: list[float] = []

    for row in feedback_rows:
        query = str(row.get("q_text") or "")
        memory_id = str(row.get("m_id") or "")
        lane = str(row.get("lane") or "").strip().lower()
        if query not in baseline:
            continue
        qr = query_result(query)
        if lane == "association":
            off_ids = qr["association_off_topk"]
            on_ids = qr["association_on_topk"]
            off_full = qr["association_off_full"]
            on_full = qr["association_on_full"]
        else:
            off_ids = qr["coarse_off_topk"]
            on_ids = qr["coarse_on_topk"]
            off_full = qr["coarse_off_full"]
            on_full = qr["coarse_on_full"]
        off_rank_full = _rank_1based(off_full, memory_id)
        if off_rank_full is not None:
            on_rank_full = _rank_1based(on_full, memory_id)
            if on_rank_full is None:
                on_rank_full = len(on_full) + 1
            feedback_rank_shift_all.append(float(max(0, int(on_rank_full) - int(off_rank_full))))
        if memory_id in off_ids:
            feedback_pair_hit_off_count += 1
            off_rank_topk = _rank_1based(off_ids, memory_id)
            on_rank_topk = _rank_1based(on_ids, memory_id)
            if off_rank_topk is not None:
                feedback_rank_shift_topk.append(
                    float(max(0, int((on_rank_topk if on_rank_topk is not None else (max(1, int(top_k)) + 1))) - int(off_rank_topk)))
                )
            if memory_id not in on_ids:
                feedback_pair_removed_count += 1
        if off_ids and memory_id == off_ids[0]:
            feedback_pair_top1_hit_off_count += 1
            if not on_ids or on_ids[0] != memory_id:
                feedback_pair_top1_removed_count += 1

    random_query_count = len(random_queries)
    random_coarse_top1_changed = 0
    random_coarse_set_changed = 0
    random_coarse_order_changed = 0
    random_association_top1_changed = 0
    random_association_set_changed = 0
    random_association_order_changed = 0
    random_dual_top1_changed = 0
    random_dual_set_changed = 0
    random_dual_order_changed = 0

    for query in random_queries:
        qr = query_result(query)
        coarse_off = qr["coarse_off_topk"]
        coarse_on = qr["coarse_on_topk"]
        association_off = qr["association_off_topk"]
        association_on = qr["association_on_topk"]

        c_top1_changed = (coarse_off[:1] != coarse_on[:1])
        c_set_changed = (set(coarse_off) != set(coarse_on))
        c_order_changed = (coarse_off != coarse_on)
        a_top1_changed = (association_off[:1] != association_on[:1])
        a_set_changed = (set(association_off) != set(association_on))
        a_order_changed = (association_off != association_on)

        if c_top1_changed:
            random_coarse_top1_changed += 1
        if c_set_changed:
            random_coarse_set_changed += 1
        if c_order_changed:
            random_coarse_order_changed += 1
        if a_top1_changed:
            random_association_top1_changed += 1
        if a_set_changed:
            random_association_set_changed += 1
        if a_order_changed:
            random_association_order_changed += 1
        if c_top1_changed or a_top1_changed:
            random_dual_top1_changed += 1
        if c_set_changed or a_set_changed:
            random_dual_set_changed += 1
        if c_order_changed or a_order_changed:
            random_dual_order_changed += 1

    feedback_remove_rate = float(feedback_pair_removed_count) / float(max(1, feedback_pair_hit_off_count))
    feedback_top1_remove_rate = float(feedback_pair_top1_removed_count) / float(max(1, feedback_pair_top1_hit_off_count))
    random_dual_top1_rate = float(random_dual_top1_changed) / float(max(1, random_query_count))
    random_dual_set_rate = float(random_dual_set_changed) / float(max(1, random_query_count))
    random_dual_order_rate = float(random_dual_order_changed) / float(max(1, random_query_count))

    conflict_total = 0
    conflict_triggered = 0
    ambiguous_total = 0
    ambiguous_noop = 0
    for qr in query_result_cache.values():
        for trace_key in ("coarse_trace", "association_trace"):
            trace = qr.get(trace_key) or {}
            for item in trace.get("rows") or []:
                per_type_scores = item.get("per_type_scores") or {}
                values = sorted((float(v) for v in per_type_scores.values()), reverse=True)
                if len(values) < 2:
                    continue
                top1 = float(values[0])
                top2 = float(values[1])
                gap = max(0.0, top1 - top2)
                is_ambiguous = top1 >= float(combo_cfg.suppressor_type_conflict_activation) and gap < float(combo_cfg.suppressor_min_type_gap)
                if not is_ambiguous:
                    continue
                ambiguous_total += 1
                if not bool(item.get("suppressed")):
                    ambiguous_noop += 1
                conflict_total += 1
                if bool(item.get("suppressed")):
                    conflict_triggered += 1

    score = (
        1.4 * feedback_remove_rate
        + 0.6 * feedback_top1_remove_rate
        - 1.6 * random_dual_top1_rate
        - 1.2 * random_dual_set_rate
        - 0.2 * random_dual_order_rate
    )
    target_ok = (
        feedback_remove_rate >= 0.90
        and 0.10 <= random_dual_set_rate <= 0.20
        and 0.10 <= random_dual_top1_rate <= 0.20
    )

    return {
        "threshold": float(combo_cfg.suppressor_threshold),
        "threshold_coarse": float(combo_cfg.suppressor_threshold_coarse),
        "threshold_association": float(combo_cfg.suppressor_threshold_association),
        "alpha": float(combo_cfg.suppressor_alpha),
        "min_delta": float(combo_cfg.suppressor_min_delta),
        "min_delta_coarse": float(combo_cfg.suppressor_min_delta_coarse),
        "min_delta_association": float(combo_cfg.suppressor_min_delta_association),
        "min_margin": float(combo_cfg.suppressor_min_margin),
        "min_zscore": float(combo_cfg.suppressor_min_zscore),
        "min_zscore_coarse": float(combo_cfg.suppressor_min_zscore_coarse),
        "min_zscore_association": float(combo_cfg.suppressor_min_zscore_association),
        "min_base_relevance": float(combo_cfg.suppressor_min_base_relevance),
        "min_type_gap": float(combo_cfg.suppressor_min_type_gap),
        "type_conflict_activation": float(combo_cfg.suppressor_type_conflict_activation),
        "bias_lambda": float(combo_cfg.suppressor_bias_lambda),
        "max_drop_per_lane": int(combo_cfg.suppressor_max_drop_per_lane),
        "keep_top_per_lane": int(combo_cfg.suppressor_keep_top_per_lane),
        "feedback_query_count": len(feedback_queries),
        "feedback_pair_count": int(feedback_pair_count),
        "feedback_pair_hit_off_count": int(feedback_pair_hit_off_count),
        "feedback_pair_removed_count": int(feedback_pair_removed_count),
        "feedback_pair_removed_rate": feedback_remove_rate,
        "feedback_top1_hit_off_count": int(feedback_pair_top1_hit_off_count),
        "feedback_top1_removed_count": int(feedback_pair_top1_removed_count),
        "feedback_top1_removed_rate": feedback_top1_remove_rate,
        "feedback_rank_shift_all_count": int(len(feedback_rank_shift_all)),
        "feedback_rank_shift_all_mean": float(sum(feedback_rank_shift_all) / float(max(1, len(feedback_rank_shift_all)))),
        "feedback_rank_shift_all_p50": _percentile(feedback_rank_shift_all, 0.50),
        "feedback_rank_shift_all_p90": _percentile(feedback_rank_shift_all, 0.90),
        "feedback_rank_shift_top5_count": int(len(feedback_rank_shift_topk)),
        "feedback_rank_shift_top5_mean": float(sum(feedback_rank_shift_topk) / float(max(1, len(feedback_rank_shift_topk)))),
        "feedback_rank_shift_top5_p50": _percentile(feedback_rank_shift_topk, 0.50),
        "feedback_rank_shift_top5_p90": _percentile(feedback_rank_shift_topk, 0.90),
        "random_query_count": int(random_query_count),
        "random_coarse_top1_changed_rate": float(random_coarse_top1_changed) / float(max(1, random_query_count)),
        "random_coarse_top5_set_changed_rate": float(random_coarse_set_changed) / float(max(1, random_query_count)),
        "random_coarse_top5_order_changed_rate": float(random_coarse_order_changed) / float(max(1, random_query_count)),
        "random_association_top1_changed_rate": float(random_association_top1_changed) / float(max(1, random_query_count)),
        "random_association_top5_set_changed_rate": float(random_association_set_changed) / float(max(1, random_query_count)),
        "random_association_top5_order_changed_rate": float(random_association_order_changed) / float(max(1, random_query_count)),
        "random_dual_top1_changed_rate": random_dual_top1_rate,
        "random_dual_top5_set_changed_rate": random_dual_set_rate,
        "random_dual_top5_order_changed_rate": random_dual_order_rate,
        "cross_type_conflict_count": int(conflict_total),
        "cross_type_conflict_trigger_count": int(conflict_triggered),
        "cross_type_conflict_trigger_rate": float(conflict_triggered) / float(max(1, conflict_total)),
        "ambiguous_case_count": int(ambiguous_total),
        "ambiguous_case_noop_count": int(ambiguous_noop),
        "ambiguous_case_noop_rate": float(ambiguous_noop) / float(max(1, ambiguous_total)),
        "target_ok": bool(target_ok),
        "score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune suppressor with feedback cohort and random cohort metrics.")
    parser.add_argument("--feedback-jsonl", default="_V5/chat/data/feedback/feedback_events.jsonl")
    parser.add_argument("--concepts-jsonl", default="data/V5/association/concepts.jsonl")
    parser.add_argument("--artifact-dir", default="data/V5/suppressor_newfb")
    parser.add_argument("--random-sample-size", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-extra", type=int, default=8)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60")
    parser.add_argument("--thresholds-coarse", default="")
    parser.add_argument("--thresholds-association", default="")
    parser.add_argument("--alphas", default="0.20,0.25,0.30")
    parser.add_argument("--min-deltas", default="0.08,0.10,0.12")
    parser.add_argument("--min-deltas-coarse", default="")
    parser.add_argument("--min-deltas-association", default="")
    parser.add_argument("--min-margins", default="0.08,0.10,0.12")
    parser.add_argument("--min-zscores", default="0.8,1.0,1.2")
    parser.add_argument("--min-zscores-coarse", default="")
    parser.add_argument("--min-zscores-association", default="")
    parser.add_argument("--min-base-relevances", default="0.0")
    parser.add_argument("--min-type-gaps", default="0.03")
    parser.add_argument("--type-conflict-activations", default="0.14")
    parser.add_argument("--bias-lambdas", default="0.0")
    parser.add_argument("--max-drops", default="1")
    parser.add_argument("--keep-tops", default="3,4")
    parser.add_argument("--output-json", default="data/V5/suppressor_newfb/two_cohort_tuning.json")
    args = parser.parse_args()

    feedback_rows = load_feedback_rows(args.feedback_jsonl)
    if not feedback_rows:
        raise RuntimeError("no feedback rows loaded")
    feedback_queries = list(dict.fromkeys(str(row.get("q_text") or "") for row in feedback_rows if str(row.get("q_text") or "")))
    random_queries = _unique_canonical_names(
        Path(args.concepts_jsonl),
        sample_size=int(args.random_sample_size),
        seed=int(args.random_seed),
    )
    all_queries = list(dict.fromkeys(feedback_queries + random_queries))
    pool_top_k = max(1, int(args.top_k) + max(0, int(args.pool_extra)))
    baseline = _precompute_baseline(query_texts=all_queries, pool_top_k=pool_top_k)

    thresholds = _ensure_non_empty(_parse_float_list(args.thresholds), 0.55)
    thresholds_coarse = _ensure_non_empty(_parse_float_list(args.thresholds_coarse), thresholds[0])
    thresholds_association = _ensure_non_empty(_parse_float_list(args.thresholds_association), thresholds[0])
    alphas = _ensure_non_empty(_parse_float_list(args.alphas), 0.25)
    min_deltas = _ensure_non_empty(_parse_float_list(args.min_deltas), 0.10)
    min_deltas_coarse = _ensure_non_empty(_parse_float_list(args.min_deltas_coarse), min_deltas[0])
    min_deltas_association = _ensure_non_empty(_parse_float_list(args.min_deltas_association), min_deltas[0])
    min_margins = _ensure_non_empty(_parse_float_list(args.min_margins), 0.10)
    min_zscores = _ensure_non_empty(_parse_float_list(args.min_zscores), 1.0)
    min_zscores_coarse = _ensure_non_empty(_parse_float_list(args.min_zscores_coarse), min_zscores[0])
    min_zscores_association = _ensure_non_empty(_parse_float_list(args.min_zscores_association), min_zscores[0])
    min_base_relevances = _ensure_non_empty(_parse_float_list(args.min_base_relevances), 0.0)
    min_type_gaps = _ensure_non_empty(_parse_float_list(args.min_type_gaps), 0.03)
    type_conflict_activations = _ensure_non_empty(_parse_float_list(args.type_conflict_activations), 0.14)
    bias_lambdas = _ensure_non_empty(_parse_float_list(args.bias_lambdas), 0.0)
    max_drops = _ensure_non_empty(_parse_int_list(args.max_drops), 1)
    keep_tops = _ensure_non_empty(_parse_int_list(args.keep_tops), 3)

    base_cfg = load_config()
    results: list[dict] = []
    combo_index = 0
    total = (
        len(thresholds_coarse)
        * len(thresholds_association)
        * len(alphas)
        * len(min_deltas_coarse)
        * len(min_deltas_association)
        * len(min_margins)
        * len(min_zscores_coarse)
        * len(min_zscores_association)
        * len(min_base_relevances)
        * len(min_type_gaps)
        * len(type_conflict_activations)
        * len(bias_lambdas)
        * len(max_drops)
        * len(keep_tops)
    )

    artifact_manifest = {}
    manifest_path = Path(args.artifact_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact_manifest = {
                "model_type": loaded_manifest.get("model_type"),
                "objective": loaded_manifest.get("objective"),
                "slots_k": loaded_manifest.get("slots_k"),
                "top_r": loaded_manifest.get("top_r"),
                "tau": loaded_manifest.get("tau"),
                "enable_memory_bias": loaded_manifest.get("enable_memory_bias"),
                "lambda_memory_bias": loaded_manifest.get("lambda_memory_bias"),
            }
        except Exception:
            artifact_manifest = {}

    for threshold_coarse in thresholds_coarse:
        for threshold_association in thresholds_association:
            for alpha in alphas:
                for min_delta_coarse in min_deltas_coarse:
                    for min_delta_association in min_deltas_association:
                        for min_margin in min_margins:
                            for min_zscore_coarse in min_zscores_coarse:
                                for min_zscore_association in min_zscores_association:
                                    for min_base_relevance in min_base_relevances:
                                        for min_type_gap in min_type_gaps:
                                            for type_conflict_activation in type_conflict_activations:
                                                for bias_lambda in bias_lambdas:
                                                    for max_drop in max_drops:
                                                        for keep_top in keep_tops:
                                                            combo_index += 1
                                                            combo_cfg = replace(
                                                                base_cfg,
                                                                suppressor_enabled=True,
                                                                suppressor_artifact_dir=Path(args.artifact_dir),
                                                                suppressor_threshold=float(max(threshold_coarse, threshold_association)),
                                                                suppressor_threshold_coarse=float(threshold_coarse),
                                                                suppressor_threshold_association=float(threshold_association),
                                                                suppressor_alpha=float(alpha),
                                                                suppressor_min_delta=float(min_delta_coarse),
                                                                suppressor_min_delta_coarse=float(min_delta_coarse),
                                                                suppressor_min_delta_association=float(min_delta_association),
                                                                suppressor_min_margin=float(min_margin),
                                                                suppressor_min_zscore=float(max(min_zscore_coarse, min_zscore_association)),
                                                                suppressor_min_zscore_coarse=float(min_zscore_coarse),
                                                                suppressor_min_zscore_association=float(min_zscore_association),
                                                                suppressor_min_base_relevance=float(min_base_relevance),
                                                                suppressor_min_type_gap=float(min_type_gap),
                                                                suppressor_type_conflict_activation=float(type_conflict_activation),
                                                                suppressor_max_drop_per_lane=int(max_drop),
                                                                suppressor_keep_top_per_lane=int(keep_top),
                                                                suppressor_extra_candidates=int(args.pool_extra),
                                                                suppressor_bias_lambda=float(bias_lambda),
                                                            )
                                                            row = _evaluate_combo(
                                                                combo_cfg=combo_cfg,
                                                                baseline=baseline,
                                                                feedback_rows=feedback_rows,
                                                                feedback_queries=feedback_queries,
                                                                random_queries=random_queries,
                                                                top_k=max(1, int(args.top_k)),
                                                            )
                                                            results.append(row)
                                                            print(
                                                                "[combo]",
                                                                f"{combo_index}/{total}",
                                                                {
                                                                    "thr_c": threshold_coarse,
                                                                    "thr_a": threshold_association,
                                                                    "alpha": alpha,
                                                                    "d_c": min_delta_coarse,
                                                                    "d_a": min_delta_association,
                                                                    "m": min_margin,
                                                                    "z_c": min_zscore_coarse,
                                                                    "z_a": min_zscore_association,
                                                                    "base_rel": min_base_relevance,
                                                                    "type_gap": min_type_gap,
                                                                    "type_act": type_conflict_activation,
                                                                    "bias": bias_lambda,
                                                                    "keep": keep_top,
                                                                    "feedback_removed": round(float(row["feedback_pair_removed_rate"]), 4),
                                                                    "feedback_shift": round(float(row["feedback_rank_shift_top5_mean"]), 4),
                                                                    "rand_top1": round(float(row["random_dual_top1_changed_rate"]), 4),
                                                                    "rand_set": round(float(row["random_dual_top5_set_changed_rate"]), 4),
                                                                    "conflict_trigger": round(float(row["cross_type_conflict_trigger_rate"]), 4),
                                                                    "amb_noop": round(float(row["ambiguous_case_noop_rate"]), 4),
                                                                    "score": round(float(row["score"]), 4),
                                                                },
                                                            )

    target_rows = [row for row in results if bool(row.get("target_ok"))]
    if target_rows:
        target_rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        best = target_rows[0]
    else:
        results.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        best = results[0] if results else {}

    payload = {
        "feedback_jsonl": str(args.feedback_jsonl),
        "concepts_jsonl": str(args.concepts_jsonl),
        "artifact_dir": str(args.artifact_dir),
        "artifact_manifest": artifact_manifest,
        "top_k": int(args.top_k),
        "pool_top_k": int(pool_top_k),
        "feedback_query_count": len(feedback_queries),
        "feedback_pair_count": len(feedback_rows),
        "random_sample_size": len(random_queries),
        "grid": {
            "thresholds": thresholds,
            "thresholds_coarse": thresholds_coarse,
            "thresholds_association": thresholds_association,
            "alphas": alphas,
            "min_deltas_coarse": min_deltas_coarse,
            "min_deltas_association": min_deltas_association,
            "min_margins": min_margins,
            "min_zscores": min_zscores,
            "min_zscores_coarse": min_zscores_coarse,
            "min_zscores_association": min_zscores_association,
            "min_base_relevances": min_base_relevances,
            "min_type_gaps": min_type_gaps,
            "type_conflict_activations": type_conflict_activations,
            "bias_lambdas": bias_lambdas,
            "max_drops": max_drops,
            "keep_tops": keep_tops,
        },
        "combo_count": len(results),
        "target_ok_count": len(target_rows),
        "best": best,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v5][suppressor][two_cohort_tuning]",
        {
            "combo_count": len(results),
            "target_ok_count": len(target_rows),
            "best": {
                "threshold": best.get("threshold"),
                "threshold_coarse": best.get("threshold_coarse"),
                "threshold_association": best.get("threshold_association"),
                "alpha": best.get("alpha"),
                "min_delta_coarse": best.get("min_delta_coarse"),
                "min_delta_association": best.get("min_delta_association"),
                "min_margin": best.get("min_margin"),
                "min_zscore": best.get("min_zscore"),
                "min_zscore_coarse": best.get("min_zscore_coarse"),
                "min_zscore_association": best.get("min_zscore_association"),
                "min_base_relevance": best.get("min_base_relevance"),
                "min_type_gap": best.get("min_type_gap"),
                "type_conflict_activation": best.get("type_conflict_activation"),
                "bias_lambda": best.get("bias_lambda"),
                "max_drop_per_lane": best.get("max_drop_per_lane"),
                "keep_top_per_lane": best.get("keep_top_per_lane"),
                "feedback_pair_removed_rate": best.get("feedback_pair_removed_rate"),
                "feedback_rank_shift_top5_mean": best.get("feedback_rank_shift_top5_mean"),
                "random_dual_top1_changed_rate": best.get("random_dual_top1_changed_rate"),
                "random_dual_top5_set_changed_rate": best.get("random_dual_top5_set_changed_rate"),
                "random_dual_top5_order_changed_rate": best.get("random_dual_top5_order_changed_rate"),
                "cross_type_conflict_trigger_rate": best.get("cross_type_conflict_trigger_rate"),
                "ambiguous_case_noop_rate": best.get("ambiguous_case_noop_rate"),
                "target_ok": best.get("target_ok"),
                "score": best.get("score"),
            },
            "output_json": str(output_path),
        },
    )


if __name__ == "__main__":
    main()
