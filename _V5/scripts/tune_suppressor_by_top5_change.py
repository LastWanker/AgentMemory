from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


V5_ROOT = Path(__file__).resolve().parents[1]
CHAT_SRC = V5_ROOT / "chat" / "src"
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))
if str(CHAT_SRC) not in sys.path:
    sys.path.insert(0, str(CHAT_SRC))

from agentmemory_v3.suppressor.data_utils import load_feedback_rows
from chat_app.config import load_config
from chat_app.models import MemoryRef
from chat_app.retriever_adapter import RetrieverAdapter
from chat_app.suppressor_adapter import SuppressorAdapter


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for chunk in str(value or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for chunk in str(value or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def _ids(refs: list[MemoryRef]) -> list[str]:
    return [str(ref.memory_id) for ref in refs]


def _retrieve_baseline(query_texts: list[str], top_k: int) -> dict[str, dict]:
    cfg = load_config()
    cfg.suppressor_enabled = False
    retriever = RetrieverAdapter(cfg)
    out: dict[str, dict] = {}
    for query in query_texts:
        coarse_refs = retriever._retrieve_coarse(query, top_k)
        association_refs: list[MemoryRef] = []
        try:
            association_refs, _trace = retriever._retrieve_association(
                query,
                top_k=top_k,
                exclude_memory_ids={ref.memory_id for ref in coarse_refs},
            )
        except Exception:
            association_refs = []
        out[query] = {
            "coarse_refs": coarse_refs,
            "association_refs": association_refs,
            "coarse_ids": _ids(coarse_refs),
            "association_ids": _ids(association_refs),
        }
    return out


def _evaluate_combo(
    *,
    baseline: dict[str, dict],
    query_weights: Counter,
    artifact_dir: Path,
    threshold: float,
    alpha: float,
    max_drop_per_lane: int,
    keep_top_per_lane: int,
) -> dict:
    cfg = load_config()
    cfg.suppressor_enabled = True
    cfg.suppressor_artifact_dir = artifact_dir
    cfg.suppressor_threshold = float(threshold)
    cfg.suppressor_alpha = float(alpha)
    cfg.suppressor_max_drop_per_lane = int(max_drop_per_lane)
    cfg.suppressor_keep_top_per_lane = int(keep_top_per_lane)
    cfg.suppressor_apply_to_coarse = True
    cfg.suppressor_apply_to_association = True
    cfg.suppressor_debug = False
    adapter = SuppressorAdapter(cfg)

    total = len(baseline)
    weight_total = float(sum(query_weights.values()))
    coarse_changed = 0
    association_changed = 0
    dual_changed = 0
    coarse_changed_weighted = 0.0
    association_changed_weighted = 0.0
    dual_changed_weighted = 0.0
    coarse_applied = 0
    association_applied = 0
    coarse_nonempty = 0
    association_nonempty = 0
    query_samples: list[dict] = []

    for idx, (query, row) in enumerate(baseline.items()):
        coarse_before = list(row["coarse_refs"])
        association_before = list(row["association_refs"])
        coarse_before_ids = list(row["coarse_ids"])
        association_before_ids = list(row["association_ids"])
        coarse_after, coarse_trace = adapter.apply(query, "coarse", coarse_before)
        association_after, association_trace = adapter.apply(query, "association", association_before)
        coarse_after_ids = _ids(coarse_after)
        association_after_ids = _ids(association_after)
        coarse_is_changed = coarse_before_ids != coarse_after_ids
        association_is_changed = association_before_ids != association_after_ids
        dual_is_changed = coarse_is_changed or association_is_changed
        weight = float(query_weights.get(query, 1))

        if coarse_before_ids:
            coarse_nonempty += 1
        if association_before_ids:
            association_nonempty += 1
        if coarse_is_changed:
            coarse_changed += 1
            coarse_changed_weighted += weight
        if association_is_changed:
            association_changed += 1
            association_changed_weighted += weight
        if dual_is_changed:
            dual_changed += 1
            dual_changed_weighted += weight
        coarse_applied += int((coarse_trace or {}).get("applied_count", 0))
        association_applied += int((association_trace or {}).get("applied_count", 0))

        if idx < 12 and dual_is_changed:
            query_samples.append(
                {
                    "query": query,
                    "coarse_before": coarse_before_ids,
                    "coarse_after": coarse_after_ids,
                    "association_before": association_before_ids,
                    "association_after": association_after_ids,
                }
            )

    return {
        "threshold": float(threshold),
        "alpha": float(alpha),
        "max_drop_per_lane": int(max_drop_per_lane),
        "keep_top_per_lane": int(keep_top_per_lane),
        "query_count": total,
        "query_weight_total": weight_total,
        "coarse_nonempty_query_count": coarse_nonempty,
        "association_nonempty_query_count": association_nonempty,
        "coarse_topk_order_changed_rate": float(coarse_changed) / float(max(1, total)),
        "association_topk_order_changed_rate": float(association_changed) / float(max(1, total)),
        "dual_topk_order_changed_rate": float(dual_changed) / float(max(1, total)),
        "coarse_topk_order_changed_rate_weighted": coarse_changed_weighted / float(max(1.0, weight_total)),
        "association_topk_order_changed_rate_weighted": association_changed_weighted / float(max(1.0, weight_total)),
        "dual_topk_order_changed_rate_weighted": dual_changed_weighted / float(max(1.0, weight_total)),
        "coarse_avg_applied_per_query": float(coarse_applied) / float(max(1, total)),
        "association_avg_applied_per_query": float(association_applied) / float(max(1, total)),
        "samples": query_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune suppressor params by dual-lane top-k order change rate.")
    parser.add_argument("--feedback-jsonl", default="_V5/chat/data/feedback/feedback_events.jsonl")
    parser.add_argument("--artifact-dir", default="data/V5/suppressor_newfb")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--alphas", default="0.25,0.35,0.45")
    parser.add_argument("--max-drops", default="1")
    parser.add_argument("--keep-tops", default="2,3")
    parser.add_argument("--output-json", default="data/V5/suppressor_newfb/top5_change_eval.json")
    args = parser.parse_args()

    feedback_rows = load_feedback_rows(args.feedback_jsonl)
    if not feedback_rows:
        raise RuntimeError("no feedback rows available")
    query_weights: Counter = Counter(str(row.get("q_text") or "") for row in feedback_rows if str(row.get("q_text") or ""))
    query_texts = list(query_weights.keys())
    if not query_texts:
        raise RuntimeError("no query texts extracted from feedback rows")

    baseline = _retrieve_baseline(query_texts=query_texts, top_k=max(1, int(args.top_k)))

    thresholds = _parse_float_list(args.thresholds)
    alphas = _parse_float_list(args.alphas)
    max_drops = _parse_int_list(args.max_drops)
    keep_tops = _parse_int_list(args.keep_tops)
    results: list[dict] = []
    for threshold in thresholds:
        for alpha in alphas:
            for max_drop in max_drops:
                for keep_top in keep_tops:
                    result = _evaluate_combo(
                        baseline=baseline,
                        query_weights=query_weights,
                        artifact_dir=Path(args.artifact_dir),
                        threshold=threshold,
                        alpha=alpha,
                        max_drop_per_lane=max_drop,
                        keep_top_per_lane=keep_top,
                    )
                    results.append(result)

    results.sort(
        key=lambda row: (
            float(row.get("dual_topk_order_changed_rate_weighted", 0.0)),
            float(row.get("association_topk_order_changed_rate_weighted", 0.0)),
            -float(row.get("coarse_avg_applied_per_query", 0.0)) - float(row.get("association_avg_applied_per_query", 0.0)),
        ),
        reverse=True,
    )
    best = results[0] if results else {}

    payload = {
        "feedback_jsonl": str(args.feedback_jsonl),
        "artifact_dir": str(args.artifact_dir),
        "query_count": len(query_texts),
        "query_weight_total": int(sum(query_weights.values())),
        "top_k": int(args.top_k),
        "best": best,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v5][suppressor][top5_change]",
        {
            "query_count": len(query_texts),
            "query_weight_total": int(sum(query_weights.values())),
            "best": {
                "threshold": best.get("threshold"),
                "alpha": best.get("alpha"),
                "max_drop_per_lane": best.get("max_drop_per_lane"),
                "keep_top_per_lane": best.get("keep_top_per_lane"),
                "dual_topk_order_changed_rate": best.get("dual_topk_order_changed_rate"),
                "dual_topk_order_changed_rate_weighted": best.get("dual_topk_order_changed_rate_weighted"),
                "association_topk_order_changed_rate": best.get("association_topk_order_changed_rate"),
                "coarse_topk_order_changed_rate": best.get("coarse_topk_order_changed_rate"),
            },
            "output_json": str(output_path),
        },
    )


if __name__ == "__main__":
    main()
