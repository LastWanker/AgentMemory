from __future__ import annotations

import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = V5_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agentmemory_v3.mem_network_suppressor import MemNetworkRuntime, MemNetworkRuntimeConfig

from .config import ChatAppConfig
from .models import MemoryRef


class MemNetworkSuppressorBackend:
    """New suppressor backend entry reserved for memory-network implementation."""

    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._runtime: MemNetworkRuntime | None = None

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
            base_trace["status"] = "mem_network_artifact_missing"
            return refs, base_trace

        cfg = MemNetworkRuntimeConfig(
            enabled=enabled,
            artifact_dir=self._config.suppressor_memnet_artifact_dir,
            score_threshold=float(self._config.suppressor_memnet_score_threshold),
            max_drop_per_lane=max(0, int(self._config.suppressor_memnet_max_drop_per_lane)),
            keep_top_per_lane=max(0, int(self._config.suppressor_memnet_keep_top_per_lane)),
            debug=bool(self._config.suppressor_debug),
        )
        rows = [
            {
                "memory_id": ref.memory_id,
                "cluster_id": ref.cluster_id,
                "score": float(ref.score),
                "base_score": float(ref.base_score if ref.base_score else ref.score),
                "source": ref.source,
                "display_text": ref.display_text,
            }
            for ref in refs
        ]
        result_rows, trace = runtime.apply_rows(
            query_text=query,
            lane=lane_key,
            rows=rows,
            config=cfg,
        )
        if result_rows is None:
            return refs, trace

        source_by_id: dict[str, MemoryRef] = {item.memory_id: item for item in refs}
        out: list[MemoryRef] = []
        for row in result_rows:
            memory_id = str(row.get("memory_id") or "")
            original = source_by_id.get(memory_id)
            if original is None:
                continue
            out.append(
                MemoryRef(
                    memory_id=memory_id,
                    cluster_id=str(row.get("cluster_id") or original.cluster_id),
                    score=float(row.get("score") if row.get("score") is not None else original.score),
                    base_score=float(row.get("base_score") if row.get("base_score") is not None else original.base_score),
                    source=str(row.get("source") or original.source),
                    display_text=str(row.get("display_text") or original.display_text),
                    suppressed=bool(row.get("suppressed") or False),
                    suppress_score=float(row.get("suppress_score") or 0.0),
                    suppress_delta=float(row.get("suppress_delta") or 0.0),
                    suppress_reason=str(row.get("suppress_reason") or ""),
                    suppress_lane=str(row.get("suppress_lane") or lane_key),
                )
            )
        if result_rows and not out:
            return refs, trace
        return out, trace

    def _get_runtime(self) -> MemNetworkRuntime | None:
        if self._runtime is not None:
            return self._runtime
        artifact_dir = self._config.suppressor_memnet_artifact_dir
        self._runtime = MemNetworkRuntime.load(str(artifact_dir) if artifact_dir else None)
        return self._runtime
