from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrieveResult:
    memory_id: str
    cluster_id: str
    score: float
    source: str
    display_text: str
