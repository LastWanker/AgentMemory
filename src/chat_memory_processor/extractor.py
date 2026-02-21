from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import UserTurn

_USER_FRAGMENT_TYPES = {"REQUEST", "USER", "HUMAN"}


def extract_user_turns(path: Path, *, session_id: Optional[str] = None) -> List[UserTurn]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    turns: List[UserTurn] = []
    if isinstance(payload, list):
        for convo_index, convo in enumerate(payload, start=1):
            if not isinstance(convo, dict):
                continue
            sid = session_id or str(convo.get("id") or f"session_{convo_index}")
            turns.extend(_extract_from_conversation(convo, session_id=sid))
        return turns

    if isinstance(payload, dict):
        sid = session_id or str(payload.get("id") or "session_1")
        turns.extend(_extract_from_conversation(payload, session_id=sid))
    return turns


def _extract_from_conversation(convo: Dict[str, object], *, session_id: str) -> List[UserTurn]:
    mapping = convo.get("mapping")
    if not isinstance(mapping, dict):
        return []

    rows: List[Tuple[str, str, str, str]] = []
    for node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        ts = str(message.get("inserted_at") or convo.get("updated_at") or "")
        for idx, text in enumerate(_iter_user_fragments(node), start=1):
            rows.append((ts, str(node_id), str(idx), text))

    rows.sort(key=lambda x: (x[0], _safe_int(x[1]), _safe_int(x[2]), x[1], x[2]))

    turns: List[UserTurn] = []
    for turn_idx, (ts, node_id, frag_idx, text) in enumerate(rows, start=1):
        turns.append(
            UserTurn(
                session_id=session_id,
                turn_id=str(turn_idx),
                role="user",
                text=text,
                timestamp=ts,
                source_node_id=f"{node_id}:{frag_idx}",
            )
        )
    return turns


def _iter_user_fragments(node: Dict[str, object]) -> Iterable[str]:
    message = node.get("message")
    if not isinstance(message, dict):
        return
    fragments = message.get("fragments")
    if not isinstance(fragments, list):
        return
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        fragment_type = str(fragment.get("type") or "").strip().upper()
        if fragment_type not in _USER_FRAGMENT_TYPES:
            continue
        content = str(fragment.get("content") or "").strip()
        if content:
            yield content


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0

