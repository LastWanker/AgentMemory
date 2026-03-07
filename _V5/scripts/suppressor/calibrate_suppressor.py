from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_v5_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src").exists() and (parent / "chat" / "src").exists():
            return parent
    return here.parents[1]


V5_ROOT = _resolve_v5_root()
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))

from agentmemory_v3.suppressor.artifacts import SuppressorArtifacts
from agentmemory_v3.suppressor.features import build_feature_vector
from agentmemory_v3.utils.io import read_jsonl


def _forward_logit(artifacts: SuppressorArtifacts, feature_vec: np.ndarray, feedback_type: str) -> float:
    tensor = torch.from_numpy(np.asarray(feature_vec, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        model_type = str(artifacts.manifest.model_type or "").strip().lower()
        if model_type == "oreo_type_heads_mlp_v1":
            logits = artifacts.model.forward_logits(tensor, feedback_type=str(feedback_type or "").strip().lower())
        else:
            logits = artifacts.model.forward_logits(tensor)
    return float(logits.reshape(-1)[0].item())


def _fit_temperature_bias(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    max_iter: int,
    min_temp: float,
    max_temp: float,
    max_abs_bias: float,
    l2_reg: float,
) -> tuple[float, float, dict]:
    logits = torch.from_numpy(np.asarray(logits_np, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(labels_np, dtype=np.float32))
    nll_before = float(F.binary_cross_entropy_with_logits(logits, labels).item())

    raw_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    raw_bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    min_t = max(1e-3, float(min_temp))
    max_t = max(min_t + 1e-3, float(max_temp))
    span_t = max_t - min_t
    bias_cap = max(0.01, float(max_abs_bias))
    reg = max(0.0, float(l2_reg))
    optimizer = torch.optim.LBFGS([raw_temp, raw_bias], lr=0.3, max_iter=max(20, int(max_iter)), line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = min_t + span_t * torch.sigmoid(raw_temp)
        bias_value = bias_cap * torch.tanh(raw_bias)
        calibrated_logits = logits / temperature + bias_value
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels)
        if reg > 0:
            loss = loss + reg * (raw_temp.pow(2).mean() + raw_bias.pow(2).mean())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = float((min_t + span_t * torch.sigmoid(raw_temp)).item())
        bias_value = float((bias_cap * torch.tanh(raw_bias)).item())
        calibrated_logits = logits / temperature + bias_value
        nll_after = float(F.binary_cross_entropy_with_logits(calibrated_logits, labels).item())
    return temperature, bias_value, {"nll_before": nll_before, "nll_after": nll_after}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-head/per-lane temperature+bias calibration for suppressor artifact.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--dataset", required=True, help="jsonl with q_text/m_id/m_text/lane/feedback_type/label")
    parser.add_argument("--output-json", default="", help="default: <artifact-dir>/calibration.json")
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--min-temp", type=float, default=0.25)
    parser.add_argument("--max-temp", type=float, default=4.0)
    parser.add_argument("--max-abs-bias", type=float, default=4.0)
    parser.add_argument("--l2-reg", type=float, default=0.002)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.output_json) if str(args.output_json).strip() else (artifact_dir / "calibration.json")
    artifacts = SuppressorArtifacts.load(artifact_dir)
    rows = list(read_jsonl(Path(args.dataset)))
    if not rows:
        raise RuntimeError("dataset is empty")

    query_cache: dict[str, np.ndarray] = {}
    missing_memory_cache: dict[str, np.ndarray] = {}

    def query_vec(text: str) -> np.ndarray:
        key = str(text or "")
        cached = query_cache.get(key)
        if cached is not None:
            return cached
        vec = artifacts.encoder.encode_query_texts([key])[0]
        query_cache[key] = vec
        return vec

    def memory_vec(memory_id: str, memory_text: str) -> np.ndarray:
        idx = artifacts.memory_index_by_id.get(str(memory_id))
        if idx is not None:
            return artifacts.memory_matrix[int(idx)]
        key = str(memory_id or memory_text or "")
        cached = missing_memory_cache.get(key)
        if cached is not None:
            return cached
        vec = artifacts.encoder.encode_passage_texts([memory_text or ""])[0]
        missing_memory_cache[key] = vec
        return vec

    grouped_lane_type: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"logits": [], "labels": []})
    grouped_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"logits": [], "labels": []})
    global_logits: list[float] = []
    global_labels: list[float] = []

    for row in rows:
        q_text = str(row.get("q_text") or "")
        m_id = str(row.get("m_id") or "")
        m_text = str(row.get("m_text") or artifacts.memory_text_by_id.get(m_id) or "")
        feedback_type = str(row.get("feedback_type") or "").strip().lower()
        lane = str(row.get("lane") or "").strip().lower()
        label = 1.0 if float(row.get("label") or 0.0) > 0.5 else 0.0
        if not feedback_type:
            continue
        feat = build_feature_vector(
            query_text=q_text,
            query_vec=query_vec(q_text),
            memory_text=m_text,
            memory_vec=memory_vec(m_id, m_text),
            feedback_type=feedback_type,
            lane=lane,
            include_query_vec=artifacts.manifest.include_query_vec,
            include_memory_vec=artifacts.manifest.include_memory_vec,
            include_cosine=artifacts.manifest.include_cosine,
            include_product=artifacts.manifest.include_product,
            include_abs_diff=artifacts.manifest.include_abs_diff,
            include_feedback_type=artifacts.manifest.include_feedback_type,
            include_lane=getattr(artifacts.manifest, "include_lane", True),
            include_lexical_overlap=artifacts.manifest.include_lexical_overlap,
            include_explicit_mention=artifacts.manifest.include_explicit_mention,
        )
        logit = _forward_logit(artifacts, feat, feedback_type)
        grouped_lane_type[(lane, feedback_type)]["logits"].append(logit)
        grouped_lane_type[(lane, feedback_type)]["labels"].append(label)
        grouped_type[feedback_type]["logits"].append(logit)
        grouped_type[feedback_type]["labels"].append(label)
        global_logits.append(logit)
        global_labels.append(label)

    if not global_logits:
        raise RuntimeError("no valid rows for calibration")

    default_temp, default_bias, default_metrics = _fit_temperature_bias(
        np.asarray(global_logits, dtype=np.float32),
        np.asarray(global_labels, dtype=np.float32),
        max_iter=int(args.max_iter),
        min_temp=float(args.min_temp),
        max_temp=float(args.max_temp),
        max_abs_bias=float(args.max_abs_bias),
        l2_reg=float(args.l2_reg),
    )
    payload = {
        "version": "temperature_bias_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": str(artifacts.manifest.model_type or ""),
        "source_dataset": str(Path(args.dataset)),
        "min_samples": int(args.min_samples),
        "default": {"temperature": default_temp, "bias": default_bias},
        "by_type": {},
        "by_lane_type": {},
        "stats": {
            "global_count": len(global_logits),
            "global_positive_rate": float(np.mean(np.asarray(global_labels, dtype=np.float32))),
            "global_nll_before": default_metrics["nll_before"],
            "global_nll_after": default_metrics["nll_after"],
            "type_fitted_count": 0,
            "lane_type_fitted_count": 0,
        },
    }

    min_samples = max(10, int(args.min_samples))
    for feedback_type, item in grouped_type.items():
        logits = np.asarray(item["logits"], dtype=np.float32)
        labels = np.asarray(item["labels"], dtype=np.float32)
        if logits.size < min_samples:
            continue
        if len(np.unique(labels)) < 2:
            continue
        temperature, bias_value, metrics = _fit_temperature_bias(
            logits,
            labels,
            max_iter=int(args.max_iter),
            min_temp=float(args.min_temp),
            max_temp=float(args.max_temp),
            max_abs_bias=float(args.max_abs_bias),
            l2_reg=float(args.l2_reg),
        )
        payload["by_type"][feedback_type] = {
            "temperature": temperature,
            "bias": bias_value,
            "count": int(logits.size),
            "positive_rate": float(np.mean(labels)),
            "nll_before": metrics["nll_before"],
            "nll_after": metrics["nll_after"],
        }
        payload["stats"]["type_fitted_count"] += 1

    for (lane, feedback_type), item in grouped_lane_type.items():
        logits = np.asarray(item["logits"], dtype=np.float32)
        labels = np.asarray(item["labels"], dtype=np.float32)
        if logits.size < min_samples:
            continue
        if len(np.unique(labels)) < 2:
            continue
        temperature, bias_value, metrics = _fit_temperature_bias(
            logits,
            labels,
            max_iter=int(args.max_iter),
            min_temp=float(args.min_temp),
            max_temp=float(args.max_temp),
            max_abs_bias=float(args.max_abs_bias),
            l2_reg=float(args.l2_reg),
        )
        lane_map = payload["by_lane_type"].setdefault(lane, {})
        lane_map[feedback_type] = {
            "temperature": temperature,
            "bias": bias_value,
            "count": int(logits.size),
            "positive_rate": float(np.mean(labels)),
            "nll_before": metrics["nll_before"],
            "nll_after": metrics["nll_after"],
        }
        payload["stats"]["lane_type_fitted_count"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v5][suppressor][calibration]",
        {
            "artifact_dir": str(artifact_dir),
            "dataset": str(args.dataset),
            "output_json": str(output_path),
            "global_count": payload["stats"]["global_count"],
            "type_fitted_count": payload["stats"]["type_fitted_count"],
            "lane_type_fitted_count": payload["stats"]["lane_type_fitted_count"],
            "global_nll_before": payload["stats"]["global_nll_before"],
            "global_nll_after": payload["stats"]["global_nll_after"],
        },
    )


if __name__ == "__main__":
    main()
