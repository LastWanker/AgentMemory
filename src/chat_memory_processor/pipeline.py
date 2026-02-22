from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .cleaning import clean_and_flag
from .clustering import assign_global_clusters
from .extractor import extract_user_turns
from .models import UserTurn
from .segmentation import assign_local_clusters


@dataclass
class ProcessorConfig:
    session_id: Optional[str] = None
    dedup_min_chars: int = 8
    segmentation_mode: str = "adaptive"
    fixed_sim_threshold: float = 0.35
    novelty_threshold: float = 0.80
    min_segment_len: int = 2
    cross_session_merge: bool = True
    global_merge_similarity_threshold: float | None = None


@dataclass
class PipelineOutput:
    raw_rows: List[Dict[str, object]]
    dedup_rows: List[Dict[str, object]]
    memory_rows: List[Dict[str, object]]


def build_processed_turns(input_path: Path, *, config: ProcessorConfig) -> PipelineOutput:
    turns = extract_user_turns(input_path, session_id=config.session_id)

    for turn in turns:
        cleaned, reason = clean_and_flag(turn.text)
        turn.text = cleaned
        if reason:
            turn.is_excluded = True
            turn.exclude_reason = reason

    by_session: Dict[str, List[UserTurn]] = defaultdict(list)
    for turn in turns:
        by_session[turn.session_id].append(turn)

    for session_turns in by_session.values():
        assign_local_clusters(
            session_turns,
            mode=config.segmentation_mode,
            fixed_sim_threshold=config.fixed_sim_threshold,
            novelty_threshold=config.novelty_threshold,
            min_segment_len=config.min_segment_len,
        )

    assign_global_clusters(
        turns,
        enable_cross_session=config.cross_session_merge,
        merge_similarity_threshold=config.global_merge_similarity_threshold,
    )

    raw_rows = [_to_row(turn, turns) for turn in turns]
    dedup_turns = _build_dedup_turns(turns, dedup_min_chars=config.dedup_min_chars)
    dedup_rows = [_to_row(turn, dedup_turns) for turn in dedup_turns]
    memory_rows = _build_memory_rows(dedup_turns)
    return PipelineOutput(
        raw_rows=raw_rows,
        dedup_rows=dedup_rows,
        memory_rows=memory_rows,
    )


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dedup_turns(turns: List[UserTurn], *, dedup_min_chars: int) -> List[UserTurn]:
    seen: Dict[tuple[str, str], str] = {}
    out: List[UserTurn] = []
    for turn in turns:
        if turn.is_excluded:
            continue
        key_text = turn.text.strip().lower()
        if len(key_text) >= dedup_min_chars:
            key = (turn.session_id, key_text)
            if key in seen:
                continue
            seen[key] = turn.turn_id
        out.append(turn)
    return out


def _build_memory_rows(turns: List[UserTurn]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, turn in enumerate(turns, start=1):
        mem_id = f"c{idx:06d}"
        rows.append(
            {
                "mem_id": mem_id,
                "text": turn.text,
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "cluster_id": turn.global_cluster_id or turn.local_cluster_id,
                "local_cluster_id": turn.local_cluster_id,
                "source_node_id": turn.source_node_id,
                "timestamp": turn.timestamp,
                "tags": ["chat", "user"],
                "source": "chat_memory_processor",
            }
        )
    return rows


def _to_row(turn: UserTurn, turns: List[UserTurn]) -> Dict[str, object]:
    cluster_id = turn.global_cluster_id or turn.local_cluster_id or turn.cluster_id
    pool_ids = [
        x.turn_id
        for x in turns
        if x.session_id == turn.session_id
        and (x.global_cluster_id or x.local_cluster_id or x.cluster_id) == cluster_id
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
        "cluster_id": cluster_id,
        "local_cluster_id": turn.local_cluster_id or turn.cluster_id,
        "global_cluster_id": turn.global_cluster_id or "",
        "candidate_pool_ids": pool_ids,
        "sim_to_prev": turn.sim_to_prev,
        "is_excluded": turn.is_excluded,
        "exclude_reason": turn.exclude_reason,
    }
