from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserTurn:
    session_id: str
    turn_id: str
    role: str
    text: str
    timestamp: str
    source_node_id: str
    topic_break_flag: bool = False
    cluster_id: str = ""
    is_excluded: bool = False
    exclude_reason: str = ""
    sim_to_prev: Optional[float] = None

