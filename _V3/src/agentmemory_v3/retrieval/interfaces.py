from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrieveRequest:
    query: str
    top_k: int = 5


@dataclass
class RetrieveResult:
    memory_id: str
    cluster_id: str
    score: float
    source: str
    display_text: str
    slot_texts: dict = field(default_factory=dict)
