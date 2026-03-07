from __future__ import annotations

import numpy as np
import torch

from .artifacts import MemNetworkArtifacts
from .config import MemNetworkRuntimeConfig
from .features import (
    aggregate_feedback_context,
    build_feature_vector,
    build_feedback_memory_slots,
    memory_id_to_index,
)
from .model import LegacyMLPScorer, MemNetworkScorer


class MemNetworkRuntime:
    def __init__(self, artifacts: MemNetworkArtifacts) -> None:
        self.artifacts = artifacts
        self._query_cache: dict[str, np.ndarray] = {}
        self._missing_memory_vec_cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, artifact_dir: str | None) -> "MemNetworkRuntime | None":
        if not artifact_dir:
            return None
        try:
            return cls(MemNetworkArtifacts.load(artifact_dir))
        except Exception:
            return None

    def apply_rows(
        self,
        *,
        query_text: str,
        lane: str,
        rows: list[dict],
        config: MemNetworkRuntimeConfig,
    ) -> tuple[list[dict], dict]:
        if self.artifacts.is_legacy_model:
            return self._apply_rows_legacy(query_text=query_text, lane=lane, rows=rows, config=config)
        return self._apply_rows_memnet(query_text=query_text, lane=lane, rows=rows, config=config)

    def _apply_rows_memnet(
        self,
        *,
        query_text: str,
        lane: str,
        rows: list[dict],
        config: MemNetworkRuntimeConfig,
    ) -> tuple[list[dict], dict]:
        lane_key = str(lane or "").strip().lower()
        if not rows:
            return rows, {
                "enabled": bool(config.enabled),
                "status": "ok",
                "model_type": str(self.artifacts.manifest.model_type),
                "lane": lane_key,
                "candidate_count": 0,
                "applied_count": 0,
                "rows": [],
            }

        query_vec = self._encode_query(query_text)
        score_threshold = float(config.score_threshold)
        feedback_top_k = max(1, int(self.artifacts.manifest.feedback_top_k))

        scored_rows: list[dict] = []
        for row in rows:
            memory_id = str(row.get("memory_id") or "")
            display_text = str(row.get("display_text") or self.artifacts.memory_text_by_id.get(memory_id) or "")
            base_score = float(row.get("base_score") if row.get("base_score") is not None else row.get("score") or 0.0)
            candidate_vec = self._get_memory_vec(memory_id, display_text)
            candidate_memory_id_id = memory_id_to_index(memory_id, self.artifacts.suppress_memory_id_to_index)
            slots = build_feedback_memory_slots(
                query_vec=query_vec,
                candidate_vec=candidate_vec,
                feedback_query_matrix=self.artifacts.feedback_query_matrix,
                feedback_memory_matrix=self.artifacts.feedback_memory_matrix,
                feedback_type_ids=self.artifacts.feedback_type_ids,
                feedback_lane_ids=self.artifacts.feedback_lane_ids,
                feedback_memory_id_ids=self.artifacts.feedback_memory_id_ids,
                candidate_memory_id_id=candidate_memory_id_id,
                top_k=feedback_top_k,
            )
            head_scores = self._predict_scores(
                query_vec=query_vec,
                candidate_vec=candidate_vec,
                candidate_memory_id_id=candidate_memory_id_id,
                slots=slots,
            )
            suppress_score = float(max(head_scores)) if head_scores else 0.0
            scored_rows.append(
                {
                    "memory_id": memory_id,
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "source": str(row.get("source") or ""),
                    "display_text": display_text,
                    "base_score": base_score,
                    "score": base_score,
                    "suppress_score": suppress_score,
                    "suppress_head_scores": [float(v) for v in head_scores],
                    "suppress_threshold": score_threshold,
                    "suppress_reason": "mem_network",
                    "suppress_lane": lane_key,
                    "suppressed": False,
                    "suppress_delta": 0.0,
                    "final_score": base_score,
                    "agg_top_indices": list(slots.get("top_indices") or []),
                    "agg_top_weights": list(slots.get("top_weights") or []),
                    "candidate_memory_id_id": int(candidate_memory_id_id),
                }
            )

        # Hard-remove policy: any candidate crossing threshold is removed immediately.
        chosen = {
            idx
            for idx, item in enumerate(scored_rows)
            if float(item.get("suppress_score") or 0.0) >= score_threshold
        }
        applied = int(len(chosen))
        out_rows: list[dict] = []
        trace_rows: list[dict] = []
        for idx, item in enumerate(scored_rows):
            removed = idx in chosen
            row_trace = {
                "memory_id": str(item.get("memory_id") or ""),
                "base_score": float(item.get("base_score") or 0.0),
                "score": float(item.get("score") or 0.0),
                "suppress_score": float(item.get("suppress_score") or 0.0),
                "suppress_head_scores": [float(v) for v in (item.get("suppress_head_scores") or [])],
                "suppressed": bool(removed),
                "removed": bool(removed),
                "suppress_delta": float(item.get("suppress_delta") or 0.0),
                "agg_top_indices": list(item.get("agg_top_indices") or []),
                "agg_top_weights": list(item.get("agg_top_weights") or []),
                "candidate_memory_id_id": int(item.get("candidate_memory_id_id") or 0),
            }
            trace_rows.append(row_trace)
            if removed:
                continue
            row_out = dict(item)
            row_out["suppressed"] = False
            row_out["suppress_delta"] = 0.0
            row_out["final_score"] = float(row_out.get("score") or 0.0)
            out_rows.append(row_out)

        trace = {
            "enabled": bool(config.enabled),
            "status": "ok",
            "model_type": str(self.artifacts.manifest.model_type),
            "lane": lane_key,
            "candidate_count": len(rows),
            "applied_count": applied,
            "hard_remove": True,
            "config": {
                "score_threshold": score_threshold,
                "max_drop_per_lane": 0,
                "keep_top_per_lane": 0,
                "suppress_strength": 0.0,
                "feedback_top_k": feedback_top_k,
            },
            "rows": trace_rows,
        }
        return out_rows, trace

    def _apply_rows_legacy(
        self,
        *,
        query_text: str,
        lane: str,
        rows: list[dict],
        config: MemNetworkRuntimeConfig,
    ) -> tuple[list[dict], dict]:
        lane_key = str(lane or "").strip().lower()
        if not rows:
            return rows, {
                "enabled": bool(config.enabled),
                "status": "ok",
                "model_type": str(self.artifacts.manifest.model_type),
                "lane": lane_key,
                "candidate_count": 0,
                "applied_count": 0,
                "rows": [],
            }
        query_vec = self._encode_query(query_text)
        score_threshold = float(config.score_threshold)
        max_drop = max(0, int(config.max_drop_per_lane))
        keep_top = max(0, int(config.keep_top_per_lane))
        suppress_strength = max(0.0, min(1.0, float(self.artifacts.manifest.suppress_strength)))
        feedback_top_k = max(1, int(self.artifacts.manifest.feedback_top_k))

        scored_rows: list[dict] = []
        for row in rows:
            memory_id = str(row.get("memory_id") or "")
            display_text = str(row.get("display_text") or self.artifacts.memory_text_by_id.get(memory_id) or "")
            base_score = float(row.get("base_score") if row.get("base_score") is not None else row.get("score") or 0.0)
            candidate_vec = self._get_memory_vec(memory_id, display_text)
            feedback_agg_vec, agg_meta = aggregate_feedback_context(
                query_vec=query_vec,
                candidate_vec=candidate_vec,
                feedback_query_matrix=self.artifacts.feedback_query_matrix,
                feedback_memory_matrix=self.artifacts.feedback_memory_matrix,
                top_k=feedback_top_k,
            )
            feat = build_feature_vector(
                query_vec=query_vec,
                candidate_vec=candidate_vec,
                feedback_agg_vec=feedback_agg_vec,
            )
            suppress_score = self._predict_score_legacy(feat)
            scored_rows.append(
                {
                    "memory_id": memory_id,
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "source": str(row.get("source") or ""),
                    "display_text": display_text,
                    "base_score": base_score,
                    "score": base_score,
                    "suppress_score": suppress_score,
                    "suppress_threshold": score_threshold,
                    "suppress_reason": "mem_network_legacy",
                    "suppress_lane": lane_key,
                    "suppressed": False,
                    "suppress_delta": 0.0,
                    "final_score": base_score,
                    "agg_top_indices": list(agg_meta.get("top_indices") or []),
                    "agg_top_weights": list(agg_meta.get("top_weights") or []),
                }
            )

        protected = {idx for idx in range(min(keep_top, len(scored_rows)))}
        eligible = [
            (idx, item)
            for idx, item in enumerate(scored_rows)
            if idx not in protected and float(item.get("suppress_score") or 0.0) >= score_threshold
        ]
        eligible.sort(
            key=lambda pair: (
                float(pair[1].get("suppress_score") or 0.0),
                float(pair[1].get("base_score") or 0.0),
            ),
            reverse=True,
        )
        chosen = {idx for idx, _ in eligible[:max_drop]}

        applied = 0
        out_rows: list[dict] = []
        for idx, item in enumerate(scored_rows):
            row_out = dict(item)
            if idx in chosen:
                new_score = float(item["base_score"]) * (1.0 - suppress_strength * float(item["suppress_score"]))
                row_out["score"] = new_score
                row_out["final_score"] = new_score
                row_out["suppress_delta"] = float(item["base_score"]) - new_score
                row_out["suppressed"] = True
                applied += 1
            out_rows.append(row_out)

        if applied > 0:
            out_rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        trace = {
            "enabled": bool(config.enabled),
            "status": "ok",
            "model_type": str(self.artifacts.manifest.model_type),
            "lane": lane_key,
            "candidate_count": len(rows),
            "applied_count": applied,
            "config": {
                "score_threshold": score_threshold,
                "max_drop_per_lane": max_drop,
                "keep_top_per_lane": keep_top,
                "suppress_strength": suppress_strength,
                "feedback_top_k": feedback_top_k,
            },
            "rows": [
                {
                    "memory_id": str(item.get("memory_id") or ""),
                    "base_score": float(item.get("base_score") or 0.0),
                    "score": float(item.get("score") or 0.0),
                    "suppress_score": float(item.get("suppress_score") or 0.0),
                    "suppressed": bool(item.get("suppressed") or False),
                    "suppress_delta": float(item.get("suppress_delta") or 0.0),
                    "agg_top_indices": list(item.get("agg_top_indices") or []),
                    "agg_top_weights": list(item.get("agg_top_weights") or []),
                }
                for item in out_rows
            ],
        }
        return out_rows, trace

    def _predict_scores(
        self,
        *,
        query_vec: np.ndarray,
        candidate_vec: np.ndarray,
        candidate_memory_id_id: int,
        slots: dict,
    ) -> list[float]:
        model = self.artifacts.model
        if not isinstance(model, MemNetworkScorer):
            return [0.0]
        with torch.no_grad():
            logits, _ = model.forward_logits(
                query_vec=torch.from_numpy(np.asarray(query_vec, dtype=np.float32)).unsqueeze(0),
                candidate_vec=torch.from_numpy(np.asarray(candidate_vec, dtype=np.float32)).unsqueeze(0),
                feedback_query_slots=torch.from_numpy(np.asarray(slots["slot_query"], dtype=np.float32)).unsqueeze(0),
                feedback_memory_slots=torch.from_numpy(np.asarray(slots["slot_memory"], dtype=np.float32)).unsqueeze(0),
                feedback_type_ids=torch.from_numpy(np.asarray(slots["slot_type_ids"], dtype=np.int64)).unsqueeze(0),
                feedback_lane_ids=torch.from_numpy(np.asarray(slots["slot_lane_ids"], dtype=np.int64)).unsqueeze(0),
                candidate_memory_id_ids=torch.from_numpy(np.asarray([int(candidate_memory_id_id)], dtype=np.int64)),
                feedback_memory_id_ids=torch.from_numpy(
                    np.asarray(slots["slot_memory_id_ids"], dtype=np.int64)
                ).unsqueeze(0),
                feedback_mask=torch.from_numpy(np.asarray(slots["slot_mask"], dtype=np.float32)).unsqueeze(0),
            )
            probs = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy().astype(np.float32)
        if probs.size == 0:
            return [0.0]
        return [float(v) for v in probs.tolist()]

    def _predict_score_legacy(self, feature_vec: np.ndarray) -> float:
        model = self.artifacts.model
        if not isinstance(model, LegacyMLPScorer):
            return 0.0
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(feature_vec, dtype=np.float32)).unsqueeze(0)
            out = model(tensor)
        return float(out.reshape(-1)[0].item())

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
