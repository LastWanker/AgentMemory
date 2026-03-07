from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuppressorRuntimeConfig:
    enabled: bool = False
    artifact_dir: Path | None = None
    threshold: float = 0.80
    alpha: float = 0.35
    min_suppress_delta: float = 0.0
    min_confidence_margin: float = 0.0
    min_zscore: float = -999.0
    min_base_relevance: float = 0.0
    min_type_gap: float = 0.0
    type_conflict_activation: float = 1.0
    max_drop_per_lane: int = 1
    keep_top_per_lane: int = 3
    apply_to_coarse: bool = True
    apply_to_association: bool = True
    debug: bool = False
    bias_lambda: float = 0.0
    use_calibration: bool = True
