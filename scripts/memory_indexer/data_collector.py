"""轻量交互日志采集器（JSONL）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class DataCollector:
    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(
        self,
        *,
        session_id: str,
        turn_index: int,
        query_text: str,
        response_text: str,
        cited_mem_ids: Optional[List[str]] = None,
        user_feedback: Optional[str] = None,
    ) -> None:
        payload: Dict[str, object] = {
            "session_id": session_id,
            "turn_index": turn_index,
            "query_text": query_text,
            "response_text": response_text,
            "cited_mem_ids": cited_mem_ids or [],
            "user_feedback": user_feedback or "",
            "timestamp": time.time(),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
