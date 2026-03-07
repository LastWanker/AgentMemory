from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemNetworkRuntimeConfig:
    enabled: bool = False
    artifact_dir: Path | None = None
    score_threshold: float = 0.5
    max_drop_per_lane: int = 1
    keep_top_per_lane: int = 3
    debug: bool = False
