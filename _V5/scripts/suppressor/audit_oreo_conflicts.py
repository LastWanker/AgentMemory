from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


V5_ROOT = Path(__file__).resolve().parents[2]
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))

from agentmemory_v3.suppressor.artifacts import SuppressorArtifacts
from agentmemory_v3.suppressor.features import build_feature_vector
from agentmemory_v3.utils.io import read_jsonl


def _read_rows(path: Path, split: str) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for row in read_jsonl(path):
        item = dict(row)
        item["_split"] = split
        out.append(item)
    return out


def _quantiles(values: list[float], points: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    return {f"q{int(round(p * 100)):02d}": float(np.quantile(arr, p)) for p in points}


def _score_summary(rows: list[dict], probs: list[float]) -> dict:
    out: dict[str, object] = {}
    labels = [int(row.get("label") or 0) for row in rows]
    pos_probs = [float(prob) for prob, label in zip(probs, labels) if label == 1]
    neg_probs = [float(prob) for prob, label in zip(probs, labels) if label == 0]
    out["overall"] = {"count": len(probs), "mean": float(np.mean(probs)) if probs else 0.0, **_quantiles(probs)}
    out["label_1"] = {"count": len(pos_probs), "mean": float(np.mean(pos_probs)) if pos_probs else 0.0, **_quantiles(pos_probs)}
    out["label_0"] = {"count": len(neg_probs), "mean": float(np.mean(neg_probs)) if neg_probs else 0.0, **_quantiles(neg_probs)}

    # Threshold sweep to expose calibration collapse around 0.5.
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.01, 0.99, 99):
        preds = [1 if float(prob) >= float(threshold) else 0 for prob in probs]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        precision = float(tp) / float(max(1, tp + fp))
        recall = float(tp) / float(max(1, tp + fn))
        f1 = 0.0 if (precision + recall) <= 1e-12 else 2.0 * precision * recall / (precision + recall)
        if f1 > float(best["f1"]):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    out["best_f1_threshold"] = best
    return out


def _build_feature_matrix(rows: list[dict], artifacts: SuppressorArtifacts) -> np.ndarray:
    manifest = artifacts.manifest
    memory_vec_by_id = artifacts.memory_index_by_id
    memory_matrix = artifacts.memory_matrix
    memory_text_by_id = artifacts.memory_text_by_id

    unique_queries = list(dict.fromkeys(str(row.get("q_text") or "") for row in rows))
    query_matrix = artifacts.encoder.encode_query_texts(unique_queries) if unique_queries else np.zeros((0, 384), dtype=np.float32)
    query_idx = {text: i for i, text in enumerate(unique_queries)}

    missing_memory_texts: dict[str, str] = {}
    for row in rows:
        m_id = str(row.get("m_id") or "")
        if m_id and m_id not in memory_vec_by_id:
            missing_memory_texts[m_id] = str(row.get("m_text") or memory_text_by_id.get(m_id) or "")
    missing_ids = list(missing_memory_texts.keys())
    missing_matrix = (
        artifacts.encoder.encode_passage_texts([missing_memory_texts[memory_id] for memory_id in missing_ids])
        if missing_ids
        else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    )
    missing_idx = {memory_id: i for i, memory_id in enumerate(missing_ids)}

    features: list[np.ndarray] = []
    for row in rows:
        q_text = str(row.get("q_text") or "")
        m_id = str(row.get("m_id") or "")
        m_text = str(row.get("m_text") or memory_text_by_id.get(m_id) or "")
        q_vec = query_matrix[int(query_idx[q_text])]
        if m_id in memory_vec_by_id:
            m_vec = memory_matrix[int(memory_vec_by_id[m_id])]
        else:
            m_vec = missing_matrix[int(missing_idx[m_id])]
        features.append(
            build_feature_vector(
                query_text=q_text,
                query_vec=q_vec,
                memory_text=m_text,
                memory_vec=m_vec,
                feedback_type=str(row.get("feedback_type") or ""),
                lane=str(row.get("lane") or ""),
                include_query_vec=bool(manifest.include_query_vec),
                include_memory_vec=bool(manifest.include_memory_vec),
                include_cosine=bool(manifest.include_cosine),
                include_product=bool(manifest.include_product),
                include_abs_diff=bool(manifest.include_abs_diff),
                include_feedback_type=bool(manifest.include_feedback_type),
                include_lane=bool(getattr(manifest, "include_lane", True)),
                include_lexical_overlap=bool(manifest.include_lexical_overlap),
                include_explicit_mention=bool(manifest.include_explicit_mention),
            )
        )
    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(features).astype(np.float32)


def _run_score_audit(rows: list[dict], artifact_dir: Path) -> dict:
    if not artifact_dir.exists():
        return {"enabled": False, "reason": "artifact_dir_not_found"}
    artifacts = SuppressorArtifacts.load(artifact_dir)
    x = _build_feature_matrix(rows, artifacts)
    if x.size == 0:
        return {"enabled": False, "reason": "empty_features"}

    with torch.no_grad():
        tensor = torch.from_numpy(x).float()
        probs = artifacts.model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)

    probs_list = [float(item) for item in probs.tolist()]
    out: dict[str, object] = {"enabled": True, "count": len(probs_list), "global": _score_summary(rows, probs_list)}

    by_type: dict[str, list[int]] = defaultdict(list)
    by_lane: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_type[str(row.get("feedback_type") or "")].append(idx)
        by_lane[str(row.get("lane") or "")].append(idx)
    out["by_feedback_type"] = {
        key: _score_summary([rows[i] for i in indices], [probs_list[i] for i in indices])
        for key, indices in sorted(by_type.items())
    }
    out["by_lane"] = {
        key: _score_summary([rows[i] for i in indices], [probs_list[i] for i in indices])
        for key, indices in sorted(by_lane.items())
    }
    return out


def _run_conflict_audit(rows: list[dict], max_examples: int) -> dict:
    summary: dict[str, object] = {}
    labels = [int(row.get("label") or 0) for row in rows]
    weights = [float(row.get("weight") or 1.0) for row in rows]
    summary["row_count"] = len(rows)
    summary["positive_count"] = int(sum(1 for label in labels if label == 1))
    summary["negative_count"] = int(sum(1 for label in labels if label == 0))
    summary["positive_rate"] = float(summary["positive_count"]) / float(max(1, len(rows)))
    weighted_pos = float(sum(weight for weight, label in zip(weights, labels) if label == 1))
    weighted_total = float(sum(weights))
    summary["weighted_positive_rate"] = weighted_pos / float(max(1e-12, weighted_total))

    by_source = Counter(str(row.get("source_kind") or "") for row in rows)
    by_type = Counter(str(row.get("feedback_type") or "") for row in rows)
    by_lane = Counter(str(row.get("lane") or "") for row in rows)
    summary["source_kind_count"] = dict(sorted(by_source.items()))
    summary["feedback_type_count"] = dict(sorted(by_type.items()))
    summary["lane_count"] = dict(sorted(by_lane.items()))

    by_source_label = defaultdict(lambda: Counter())
    by_type_label = defaultdict(lambda: Counter())
    by_lane_label = defaultdict(lambda: Counter())
    for row in rows:
        label = int(row.get("label") or 0)
        by_source_label[str(row.get("source_kind") or "")][label] += 1
        by_type_label[str(row.get("feedback_type") or "")][label] += 1
        by_lane_label[str(row.get("lane") or "")][label] += 1
    summary["source_kind_label"] = {k: {"0": int(v[0]), "1": int(v[1])} for k, v in sorted(by_source_label.items())}
    summary["feedback_type_label"] = {k: {"0": int(v[0]), "1": int(v[1])} for k, v in sorted(by_type_label.items())}
    summary["lane_label"] = {k: {"0": int(v[0]), "1": int(v[1])} for k, v in sorted(by_lane_label.items())}

    # Exact conflict: same (q,m,lane,feedback_type) got both labels.
    key_exact_map: dict[tuple[str, str, str, str], dict] = defaultdict(lambda: {"labels": set(), "count": 0, "anchors": set()})
    for row in rows:
        key = (
            str(row.get("q_text") or ""),
            str(row.get("m_id") or ""),
            str(row.get("lane") or ""),
            str(row.get("feedback_type") or ""),
        )
        key_exact_map[key]["labels"].add(int(row.get("label") or 0))
        key_exact_map[key]["count"] += 1
        key_exact_map[key]["anchors"].add(str(row.get("anchor_feedback_id") or ""))
    exact_conflicts = [(key, value) for key, value in key_exact_map.items() if len(value["labels"]) > 1]
    exact_conflict_rows = int(sum(int(value["count"]) for _, value in exact_conflicts))
    summary["exact_key_conflict"] = {
        "unique_key_count": len(key_exact_map),
        "conflict_key_count": len(exact_conflicts),
        "conflict_key_rate": float(len(exact_conflicts)) / float(max(1, len(key_exact_map))),
        "conflict_row_count": exact_conflict_rows,
        "conflict_row_rate": float(exact_conflict_rows) / float(max(1, len(rows))),
    }

    # Cross-type polarity: same (q,m,lane), unrelated and toforget pull opposite directions.
    key_cross_map: dict[tuple[str, str, str], dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        key = (str(row.get("q_text") or ""), str(row.get("m_id") or ""), str(row.get("lane") or ""))
        key_cross_map[key][str(row.get("feedback_type") or "")][int(row.get("label") or 0)] += 1

    cross_type_conflicts: list[tuple[tuple[str, str, str], dict[str, Counter]]] = []
    same_key_any_polarity: list[tuple[tuple[str, str, str], dict[str, Counter]]] = []
    for key, by_feedback in key_cross_map.items():
        has_pos = any(counter[1] > 0 for counter in by_feedback.values())
        has_neg = any(counter[0] > 0 for counter in by_feedback.values())
        if has_pos and has_neg:
            same_key_any_polarity.append((key, by_feedback))
        unrelated = by_feedback.get("unrelated", Counter())
        toforget = by_feedback.get("toforget", Counter())
        unrelated_pos = unrelated[1] > 0
        toforget_pos = toforget[1] > 0
        unrelated_neg = unrelated[0] > 0
        toforget_neg = toforget[0] > 0
        if (unrelated_pos and toforget_neg) or (toforget_pos and unrelated_neg):
            cross_type_conflicts.append((key, by_feedback))

    summary["cross_type_conflict"] = {
        "q_m_lane_key_count": len(key_cross_map),
        "any_polarity_conflict_key_count": len(same_key_any_polarity),
        "any_polarity_conflict_key_rate": float(len(same_key_any_polarity)) / float(max(1, len(key_cross_map))),
        "unrelated_vs_toforget_conflict_key_count": len(cross_type_conflicts),
        "unrelated_vs_toforget_conflict_key_rate": float(len(cross_type_conflicts)) / float(max(1, len(key_cross_map))),
    }

    # Memory-level polarity conflict.
    memory_label = defaultdict(set)
    for row in rows:
        memory_label[str(row.get("m_id") or "")].add(int(row.get("label") or 0))
    memory_conflict_count = sum(1 for labels_set in memory_label.values() if len(labels_set) > 1)
    summary["memory_level_conflict"] = {
        "memory_count": len(memory_label),
        "memory_conflict_count": int(memory_conflict_count),
        "memory_conflict_rate": float(memory_conflict_count) / float(max(1, len(memory_label))),
    }

    def _example_item(key, value) -> dict:
        has_exact_fields = isinstance(value, dict) and ("labels" in value or "count" in value or "anchors" in value)
        by_feedback_type = None
        if isinstance(value, dict) and not has_exact_fields:
            by_feedback_type = {
                feedback_type: {"0": int(counter[0]), "1": int(counter[1])}
                for feedback_type, counter in sorted(value.items())
            }
        return {
            "q_text": key[0],
            "m_id": key[1],
            "lane": key[2],
            "feedback_type": key[3] if len(key) > 3 else None,
            "labels": sorted(int(label) for label in value["labels"]) if has_exact_fields and "labels" in value else None,
            "count": int(value["count"]) if has_exact_fields and "count" in value else None,
            "anchors": sorted(anchor for anchor in (value.get("anchors") or []) if anchor)[:10]
            if has_exact_fields and isinstance(value, dict)
            else None,
            "by_feedback_type": by_feedback_type,
        }

    summary["examples"] = {
        "exact_key_conflict_examples": [_example_item(key, value) for key, value in exact_conflicts[:max(0, int(max_examples))]],
        "cross_type_conflict_examples": [_example_item(key, value) for key, value in cross_type_conflicts[:max(0, int(max_examples))]],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit suppressor sample conflicts and Oreo score calibration.")
    parser.add_argument("--data-dir", default="data/V5/suppressor_oreo_r1_dataset")
    parser.add_argument("--artifact-dir", default="data/V5/suppressor_oreo_r1_artifact")
    parser.add_argument("--output-json", default="data/V5/suppressor_oreo_r1_eval/conflict_audit_report.json")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_rows = _read_rows(data_dir / "feedback_samples_train.jsonl", split="train")
    valid_rows = _read_rows(data_dir / "feedback_samples_valid.jsonl", split="valid")
    rows = train_rows + valid_rows
    if not rows:
        raise RuntimeError("no rows found in dataset")

    conflict = _run_conflict_audit(rows, max_examples=max(1, int(args.max_examples)))
    score = _run_score_audit(rows, artifact_dir=Path(args.artifact_dir))

    payload = {
        "data_dir": str(data_dir),
        "artifact_dir": str(args.artifact_dir),
        "row_count": len(rows),
        "split_count": {"train": len(train_rows), "valid": len(valid_rows)},
        "conflict_audit": conflict,
        "score_audit": score,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v5][suppressor][oreo][audit]",
        {
            "row_count": len(rows),
            "output_json": str(output_path),
            "exact_conflict_key_count": int(((payload.get("conflict_audit") or {}).get("exact_key_conflict") or {}).get("conflict_key_count", 0)),
            "cross_type_conflict_key_count": int(((payload.get("conflict_audit") or {}).get("cross_type_conflict") or {}).get("unrelated_vs_toforget_conflict_key_count", 0)),
            "score_audit_enabled": bool((payload.get("score_audit") or {}).get("enabled")),
        },
    )


if __name__ == "__main__":
    main()
