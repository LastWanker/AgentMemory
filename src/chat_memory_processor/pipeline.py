from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cleaning import clean_and_flag
from .extractor import extract_user_turns
from .models import UserTurn
from .segmentation import assign_clusters


@dataclass
class ProcessorConfig:
    session_id: Optional[str] = None
    sim_threshold: float = 0.35
    dedup_min_chars: int = 8


def build_processed_turns(
    input_path: Path,
    *,
    config: ProcessorConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    turns = extract_user_turns(input_path, session_id=config.session_id)
    for turn in turns:
        cleaned, reason = clean_and_flag(turn.text)
        turn.text = cleaned
        if reason:
            turn.is_excluded = True
            turn.exclude_reason = reason

    # Keep segmentation strictly inside each session.
    by_session: Dict[str, List[UserTurn]] = defaultdict(list)
    for turn in turns:
        by_session[turn.session_id].append(turn)
    for session_turns in by_session.values():
        assign_clusters(session_turns, sim_threshold=config.sim_threshold)

    raw_rows = [_to_row(turn, turns) for turn in turns]
    dedup_rows = _build_dedup_rows(turns, dedup_min_chars=config.dedup_min_chars)
    return raw_rows, dedup_rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dedup_rows(turns: List[UserTurn], *, dedup_min_chars: int) -> List[Dict[str, object]]:
    seen: Dict[Tuple[str, str], str] = {}
    out: List[Dict[str, object]] = []
    for idx, turn in enumerate(turns):
        if turn.is_excluded:
            continue
        key_text = turn.text.strip().lower()
        if len(key_text) >= dedup_min_chars:
            key = (turn.session_id, key_text)
            if key in seen:
                continue
            seen[key] = turn.turn_id
        out.append(_to_row(turn, turns))
    return out


def _to_row(turn: UserTurn, turns: List[UserTurn]) -> Dict[str, object]:
    pool_ids = [
        x.turn_id
        for x in turns
        if x.session_id == turn.session_id and x.cluster_id == turn.cluster_id
    ]
    return {
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "role": turn.role,
        "text": turn.text,
        "timestamp": turn.timestamp,
        "source_node_id": turn.source_node_id,
        "user_message_clean": turn.text,
        "topic_break_flag": turn.topic_break_flag,
        "cluster_id": turn.cluster_id,
        "candidate_pool_ids": pool_ids,
        "sim_to_prev": turn.sim_to_prev,
        "is_excluded": turn.is_excluded,
        "exclude_reason": turn.exclude_reason,
    }
