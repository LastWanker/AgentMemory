from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import ChatTurn, MemoryRef


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def new_session_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]

    def session_path(self, session_id: str) -> Path:
        return self._data_dir / f"{session_id}.jsonl"

    def append_turn(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        memory_refs: list[MemoryRef] | None = None,
        metadata: dict | None = None,
    ) -> None:
        row = {
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "memory_refs": [ref.model_dump() for ref in (memory_refs or [])],
            "metadata": metadata or {},
        }
        path = self.session_path(session_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_history(self, session_id: str) -> list[ChatTurn]:
        path = self.session_path(session_id)
        if not path.exists():
            return []
        turns: list[ChatTurn] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns.append(ChatTurn.model_validate_json(line))
        return turns
