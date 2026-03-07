from __future__ import annotations

from .config import ChatAppConfig
from .models import MemoryRef
from .suppressor_backend_legacy import LegacySuppressorBackend
from .suppressor_backend_memnet import MemNetworkSuppressorBackend


class _NoopSuppressorBackend:
    def __init__(self, config: ChatAppConfig, *, status: str) -> None:
        self._config = config
        self._status = status

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
        return refs, {
            "enabled": enabled,
            "lane": lane_key,
            "status": self._status,
            "candidate_count": len(refs),
            "applied_count": 0,
            "rows": [],
        }


class SuppressorAdapter:
    """Backend router: keep retrieval path stable while allowing suppressor hot-swap."""

    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._backend_name, self._backend = self._build_backend(config)

    def apply(
        self,
        query: str,
        lane: str,
        refs: list[MemoryRef],
        *,
        enabled_override: bool | None = None,
    ) -> tuple[list[MemoryRef], dict]:
        out, trace = self._backend.apply(
            query=query,
            lane=lane,
            refs=refs,
            enabled_override=enabled_override,
        )
        if isinstance(trace, dict):
            trace["backend"] = self._backend_name
        return out, trace

    @staticmethod
    def _build_backend(config: ChatAppConfig):
        raw_backend = str(config.suppressor_backend or "").strip().lower()
        if raw_backend in {"", "off", "disabled", "none"}:
            return "off", _NoopSuppressorBackend(config, status="backend_off")
        if raw_backend in {"legacy", "oreo", "mlp"}:
            return "legacy", LegacySuppressorBackend(config)
        if raw_backend in {"memnet", "mem_network", "memory_network"}:
            return "mem_network", MemNetworkSuppressorBackend(config)
        return raw_backend, _NoopSuppressorBackend(config, status="backend_unknown")
