from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SlotPayload:
    raw_brief: str = ""
    event: str = ""
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    status: str = ""
    emotion: str = ""
    context: str = ""
    impact: str = ""


@dataclass
class MemoryRecord:
    memory_id: str
    cluster_id: str
    raw_text: str
    prev_raw: str = ""
    next_raw: str = ""


@dataclass
class QueryRecord:
    query_id: str
    text: str
    positives: List[str] = field(default_factory=list)


@dataclass
class RetrievalHit:
    memory_id: str
    cluster_id: str
    score: float
    source: str
    display_text: str
