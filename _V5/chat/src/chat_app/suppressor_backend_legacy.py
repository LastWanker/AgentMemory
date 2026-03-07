from __future__ import annotations

import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = V5_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agentmemory_v3.suppressor import SuppressorRuntime, SuppressorRuntimeConfig

from .config import ChatAppConfig
from .models import MemoryRef


class LegacySuppressorBackend:
    """Legacy MLP/Oreo suppressor backend kept for migration fallback only."""

    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._runtime: SuppressorRuntime | None = None

    def apply(
        self,
        query: str,
        lane: str,
        refs: list[MemoryRef],
        *,
        enabled_override: bool | None = None,
    ) -> tuple[list[MemoryRef], dict]:
        lane_key = str(lane or "").strip().lower()
        enabled = bool(self._config.suppressor_enabled) if enabled_override is None else bool(enabled_override)
        lane_allowed = (
            (lane_key == "coarse" and bool(self._config.suppressor_apply_to_coarse))
            or (lane_key == "association" and bool(self._config.suppressor_apply_to_association))
        )
        base_trace = {
            "enabled": enabled,
            "lane": lane_key,
            "status": "disabled",
            "candidate_count": len(refs),
            "applied_count": 0,
            "rows": [],
        }
        if not enabled or not lane_allowed or not refs:
            return refs, base_trace

        runtime = self._get_runtime()
        if runtime is None:
            base_trace["status"] = "artifact_missing"
            return refs, base_trace

        lane_threshold = (
            float(self._config.suppressor_threshold_coarse)
            if lane_key == "coarse"
            else float(self._config.suppressor_threshold_association)
        )
        lane_min_zscore = (
            float(self._config.suppressor_min_zscore_coarse)
            if lane_key == "coarse"
            else float(self._config.suppressor_min_zscore_association)
        )
        cfg = SuppressorRuntimeConfig(
            enabled=enabled,
            artifact_dir=self._config.suppressor_artifact_dir,
            threshold=lane_threshold,
            alpha=float(self._config.suppressor_alpha),
            min_suppress_delta=float(
                self._config.suppressor_min_delta_coarse
                if lane_key == "coarse"
                else self._config.suppressor_min_delta_association
            ),
            min_confidence_margin=float(self._config.suppressor_min_margin),
            min_zscore=lane_min_zscore,
            min_base_relevance=float(self._config.suppressor_min_base_relevance),
            min_type_gap=float(self._config.suppressor_min_type_gap),
            type_conflict_activation=float(self._config.suppressor_type_conflict_activation),
            max_drop_per_lane=int(self._config.suppressor_max_drop_per_lane),
            keep_top_per_lane=int(self._config.suppressor_keep_top_per_lane),
            apply_to_coarse=bool(self._config.suppressor_apply_to_coarse),
            apply_to_association=bool(self._config.suppressor_apply_to_association),
            debug=bool(self._config.suppressor_debug),
            bias_lambda=float(self._config.suppressor_bias_lambda),
            use_calibration=bool(self._config.suppressor_use_calibration),
        )
        rows = []
        source_by_id: dict[str, tuple[str, str]] = {}
        for ref in refs:
            rows.append(
                {
                    "memory_id": ref.memory_id,
                    "display_text": ref.display_text,
                    "score": float(ref.score),
                    "base_score": float(ref.base_score if ref.base_score else ref.score),
                    "lane": lane_key,
                }
            )
            source_by_id[ref.memory_id] = (ref.cluster_id, ref.source)
        scored_rows, trace = runtime.score_rows(query_text=query, rows=rows, config=cfg)
        trace["lane"] = lane_key
        trace["status"] = str(trace.get("status") or "ok")

        out: list[MemoryRef] = []
        for item in scored_rows:
            cluster_id, source = source_by_id.get(item.memory_id, ("", ""))
            out.append(
                MemoryRef(
                    memory_id=item.memory_id,
                    cluster_id=cluster_id,
                    score=float(item.final_score),
                    base_score=float(item.base_score),
                    source=source,
                    display_text=item.memory_text,
                    suppressed=bool(item.suppressed),
                    suppress_score=float(item.suppress_score),
                    suppress_delta=float(item.suppress_delta),
                    suppress_reason=str(item.feedback_type or ""),
                    suppress_lane=lane_key,
                )
            )
        return out, trace

    def _get_runtime(self) -> SuppressorRuntime | None:
        if self._runtime is not None:
            return self._runtime
        artifact_dir = self._config.suppressor_artifact_dir
        self._runtime = SuppressorRuntime.load(str(artifact_dir) if artifact_dir else None)
        return self._runtime
