from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .artifacts import SuppressorArtifacts
from .config import SuppressorRuntimeConfig
from .features import FEEDBACK_TYPES, build_feature_vector


@dataclass(frozen=True)
class SuppressorScore:
    memory_id: str
    feedback_type: str
    raw_suppress_score: float
    suppress_score: float
    type_gap: float
    type_conflict_blocked: bool
    memory_bias: float
    per_type_scores: dict[str, float]
    per_type_raw_scores: dict[str, float]
    base_score: float
    final_score: float
    suppress_delta: float
    suppressed: bool
    memory_text: str


class SuppressorRuntime:
    def __init__(self, artifacts: SuppressorArtifacts) -> None:
        self.artifacts = artifacts
        self._query_cache: dict[str, np.ndarray] = {}
        self._missing_memory_vec_cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, artifact_dir: str | None) -> "SuppressorRuntime | None":
        if not artifact_dir:
            return None
        try:
            return cls(SuppressorArtifacts.load(artifact_dir))
        except Exception:
            return None

    def score_rows(
        self,
        *,
        query_text: str,
        rows: list[dict],
        config: SuppressorRuntimeConfig,
    ) -> tuple[list[SuppressorScore], dict]:
        manifest = self.artifacts.manifest
        min_type_gap = max(0.0, float(config.min_type_gap))
        type_conflict_activation = max(0.0, float(config.type_conflict_activation))
        min_base_relevance = float(config.min_base_relevance)
        use_calibration = bool(config.use_calibration)
        bias_lambda = (
            float(config.bias_lambda)
            if float(config.bias_lambda) > 0
            else (float(manifest.lambda_memory_bias) if bool(manifest.enable_memory_bias) else 0.0)
        )
        if not rows:
            return [], {
                "enabled": bool(config.enabled),
                "model_type": manifest.model_type,
                "candidate_count": 0,
                "applied_count": 0,
                "rows": [],
            }

        query_vec = self._encode_query(query_text)
        scored_rows: list[SuppressorScore] = []
        for row in rows:
            lane_key = str(row.get("lane") or "").strip().lower()
            memory_id = str(row.get("memory_id") or "")
            memory_text = str(row.get("display_text") or self.artifacts.memory_text_by_id.get(memory_id) or "")
            base_score = float(row.get("base_score") if row.get("base_score") is not None else row.get("score") or 0.0)
            memory_vec = self._get_memory_vec(memory_id, memory_text)
            per_type_scores: dict[str, float] = {}
            per_type_raw_scores: dict[str, float] = {}
            for feedback_type in FEEDBACK_TYPES:
                feature_vec = build_feature_vector(
                    query_text=query_text,
                    query_vec=query_vec,
                    memory_text=memory_text,
                    memory_vec=memory_vec,
                    feedback_type=feedback_type,
                    lane=str(row.get("lane") or ""),
                    include_query_vec=manifest.include_query_vec,
                    include_memory_vec=manifest.include_memory_vec,
                    include_cosine=manifest.include_cosine,
                    include_product=manifest.include_product,
                    include_abs_diff=manifest.include_abs_diff,
                    include_feedback_type=manifest.include_feedback_type,
                    include_lane=getattr(manifest, "include_lane", True),
                    include_lexical_overlap=manifest.include_lexical_overlap,
                    include_explicit_mention=manifest.include_explicit_mention,
                )
                raw_logit, raw_score = self._predict_raw(feature_vec, feedback_type=feedback_type)
                cal_score = raw_score
                cal_temperature = 1.0
                cal_bias = 0.0
                if use_calibration:
                    cal_temperature, cal_bias, _cal_source = self.artifacts.get_calibration_params(
                        lane=lane_key,
                        feedback_type=feedback_type,
                    )
                    cal_logit = float(raw_logit) / max(1e-3, float(cal_temperature)) + float(cal_bias)
                    cal_score = float(torch.sigmoid(torch.tensor(cal_logit)).item())
                per_type_raw_scores[feedback_type] = raw_score
                per_type_scores[feedback_type] = cal_score
            sorted_pairs = sorted(per_type_scores.items(), key=lambda pair: float(pair[1]), reverse=True)
            best_type = str(sorted_pairs[0][0]) if sorted_pairs else ""
            best_score = float(sorted_pairs[0][1]) if sorted_pairs else 0.0
            best_raw_score = float(per_type_raw_scores.get(best_type, best_score))
            second_score = float(sorted_pairs[1][1]) if len(sorted_pairs) > 1 else 0.0
            type_gap = max(0.0, best_score - second_score)
            type_conflict_blocked = bool(
                best_score >= type_conflict_activation
                and type_gap < min_type_gap
            )
            memory_bias = float(self.artifacts.memory_bias_by_id.get(memory_id, 0.0))
            adjusted_score = 0.0
            if not type_conflict_blocked:
                adjusted_score = float(best_score) - bias_lambda * memory_bias
                adjusted_score = max(0.0, min(1.0, adjusted_score))
            scored_rows.append(
                SuppressorScore(
                    memory_id=memory_id,
                    feedback_type=(best_type if not type_conflict_blocked else "conflict_hold"),
                    raw_suppress_score=float(best_raw_score),
                    suppress_score=float(adjusted_score),
                    type_gap=float(type_gap),
                    type_conflict_blocked=bool(type_conflict_blocked),
                    memory_bias=memory_bias,
                    per_type_scores=per_type_scores,
                    per_type_raw_scores=per_type_raw_scores,
                    base_score=base_score,
                    final_score=base_score,
                    suppress_delta=0.0,
                    suppressed=False,
                    memory_text=memory_text,
                )
            )

        protected = {idx for idx in range(min(max(0, int(config.keep_top_per_lane)), len(scored_rows)))}
        suppress_scores = np.asarray([float(item.suppress_score) for item in scored_rows], dtype=np.float32)
        score_mean = float(np.mean(suppress_scores)) if suppress_scores.size else 0.0
        score_std = float(np.std(suppress_scores)) if suppress_scores.size else 0.0
        score_std = max(score_std, 1e-6)
        row_metrics: dict[int, dict] = {}
        eligible: list[tuple[int, SuppressorScore]] = []
        for idx, item in enumerate(scored_rows):
            confidence_margin = float(item.type_gap)
            predicted_delta = float(item.base_score) * float(config.alpha) * float(item.suppress_score)
            zscore = (float(item.suppress_score) - score_mean) / score_std
            passes_threshold = float(item.suppress_score) > float(config.threshold)
            passes_margin = confidence_margin >= float(config.min_confidence_margin)
            passes_delta = predicted_delta >= float(config.min_suppress_delta)
            passes_zscore = zscore >= float(config.min_zscore)
            passes_base_relevance = float(item.base_score) >= min_base_relevance
            is_protected = idx in protected
            is_eligible = (
                (not is_protected)
                and passes_threshold
                and passes_margin
                and passes_delta
                and passes_zscore
                and passes_base_relevance
            )
            row_metrics[idx] = {
                "confidence_margin": confidence_margin,
                "predicted_delta": predicted_delta,
                "zscore": zscore,
                "passes_threshold": passes_threshold,
                "passes_margin": passes_margin,
                "passes_delta": passes_delta,
                "passes_zscore": passes_zscore,
                "passes_base_relevance": passes_base_relevance,
                "protected": is_protected,
                "eligible": is_eligible,
            }
            if is_eligible:
                eligible.append((idx, item))
        eligible.sort(key=lambda pair: (float(pair[1].suppress_score), float(pair[1].base_score)), reverse=True)
        chosen = {idx for idx, _ in eligible[: max(0, int(config.max_drop_per_lane))]}

        applied_count = 0
        final_rows: list[SuppressorScore] = []
        for idx, item in enumerate(scored_rows):
            if idx in chosen:
                final_score = float(item.base_score) * (1.0 - float(config.alpha) * float(item.suppress_score))
                final_rows.append(
                    SuppressorScore(
                        memory_id=item.memory_id,
                        feedback_type=item.feedback_type,
                        raw_suppress_score=item.raw_suppress_score,
                        suppress_score=item.suppress_score,
                        type_gap=item.type_gap,
                        type_conflict_blocked=item.type_conflict_blocked,
                        memory_bias=item.memory_bias,
                        per_type_scores=item.per_type_scores,
                        per_type_raw_scores=item.per_type_raw_scores,
                        base_score=item.base_score,
                        final_score=final_score,
                        suppress_delta=float(item.base_score) - final_score,
                        suppressed=True,
                        memory_text=item.memory_text,
                    )
                )
                applied_count += 1
            else:
                final_rows.append(item)

        trace = {
            "enabled": bool(config.enabled),
            "model_type": manifest.model_type,
            "candidate_count": len(rows),
            "applied_count": applied_count,
            "config": {
                "threshold": float(config.threshold),
                "alpha": float(config.alpha),
                "min_suppress_delta": float(config.min_suppress_delta),
                "min_confidence_margin": float(config.min_confidence_margin),
                "min_zscore": float(config.min_zscore),
                "min_base_relevance": float(min_base_relevance),
                "min_type_gap": float(min_type_gap),
                "type_conflict_activation": float(type_conflict_activation),
                "max_drop_per_lane": int(config.max_drop_per_lane),
                "keep_top_per_lane": int(config.keep_top_per_lane),
                "score_mean": score_mean,
                "score_std": score_std,
                "bias_lambda": bias_lambda,
                "use_calibration": use_calibration,
            },
            "rows": [
                {
                    "memory_id": item.memory_id,
                    "feedback_type": item.feedback_type,
                    "base_score": float(item.base_score),
                    "raw_suppress_score": float(item.raw_suppress_score),
                    "suppress_score": float(item.suppress_score),
                    "type_gap": float(item.type_gap),
                    "type_conflict_blocked": bool(item.type_conflict_blocked),
                    "memory_bias": float(item.memory_bias),
                    "per_type_scores": {key: float(value) for key, value in item.per_type_scores.items()},
                    "per_type_raw_scores": {key: float(value) for key, value in item.per_type_raw_scores.items()},
                    "confidence_margin": float((row_metrics.get(idx) or {}).get("confidence_margin", 0.0)),
                    "predicted_delta": float((row_metrics.get(idx) or {}).get("predicted_delta", 0.0)),
                    "zscore": float((row_metrics.get(idx) or {}).get("zscore", 0.0)),
                    "passes_threshold": bool((row_metrics.get(idx) or {}).get("passes_threshold", False)),
                    "passes_margin": bool((row_metrics.get(idx) or {}).get("passes_margin", False)),
                    "passes_delta": bool((row_metrics.get(idx) or {}).get("passes_delta", False)),
                    "passes_zscore": bool((row_metrics.get(idx) or {}).get("passes_zscore", False)),
                    "passes_base_relevance": bool((row_metrics.get(idx) or {}).get("passes_base_relevance", False)),
                    "protected": bool((row_metrics.get(idx) or {}).get("protected", False)),
                    "eligible": bool((row_metrics.get(idx) or {}).get("eligible", False)),
                    "final_score": float(item.final_score),
                    "suppress_delta": float(item.suppress_delta),
                    "suppressed": bool(item.suppressed),
                }
                for idx, item in enumerate(final_rows)
            ],
        }
        if applied_count > 0:
            final_rows.sort(key=lambda item: float(item.final_score), reverse=True)
        return final_rows, trace

    def _predict_raw(self, feature_vec: np.ndarray, *, feedback_type: str = "") -> tuple[float, float]:
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(feature_vec, dtype=np.float32)).unsqueeze(0)
            if str(self.artifacts.manifest.model_type or "").strip().lower() == "oreo_type_heads_mlp_v1":
                logits = self.artifacts.model.forward_logits(tensor, feedback_type=str(feedback_type or "").strip().lower())
            else:
                logits = self.artifacts.model.forward_logits(tensor)
            out = torch.sigmoid(logits)
        return float(logits.reshape(-1)[0].item()), float(out.reshape(-1)[0].item())

    def _encode_query(self, query_text: str) -> np.ndarray:
        key = str(query_text or "")
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        vec = self.artifacts.encoder.encode_query_texts([key])[0]
        self._query_cache[key] = vec
        return vec

    def _get_memory_vec(self, memory_id: str, memory_text: str) -> np.ndarray:
        idx = self.artifacts.memory_index_by_id.get(str(memory_id))
        if idx is not None:
            return self.artifacts.memory_matrix[int(idx)]
        key = str(memory_id or memory_text or "")
        cached = self._missing_memory_vec_cache.get(key)
        if cached is not None:
            return cached
        vec = self.artifacts.encoder.encode_passage_texts([memory_text or ""])[0]
        self._missing_memory_vec_cache[key] = vec
        return vec
