from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class FeedbackStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._feedback_dir = base_dir / "feedback"
        self._feedback_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._feedback_dir / "feedback_events.jsonl"

    def append_feedback(
        self,
        *,
        session_id: str,
        query_text: str,
        memory_id: str,
        feedback_type: str,
        lane: str,
        candidate_refs: list[dict],
    ) -> dict:
        normalized_query = str(query_text or "").strip()
        normalized_memory_id = str(memory_id or "").strip()
        normalized_type = str(feedback_type or "").strip().lower()
        normalized_lane = str(lane or "").strip().lower()
        prior_rows = self._load_rows_for_query(session_id=session_id, query_text=normalized_query)
        for row in prior_rows:
            if str(row.get("memory_id") or "").strip() == normalized_memory_id:
                return row
        prior_self_ids = {str(row.get("memory_id") or "") for row in prior_rows}

        dedup_candidate_refs: list[dict] = []
        seen_ids: set[str] = set()
        for row in candidate_refs:
            candidate_memory_id = str(row.get("memory_id") or "").strip()
            if not candidate_memory_id or candidate_memory_id in seen_ids:
                continue
            seen_ids.add(candidate_memory_id)
            dedup_candidate_refs.append(
                {
                    "memory_id": candidate_memory_id,
                    "lane": str(row.get("lane") or "").strip().lower(),
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "score": float(row.get("score") or 0.0),
                    "base_score": float(row.get("base_score") or row.get("score") or 0.0),
                    "source": str(row.get("source") or ""),
                    "display_text": str(row.get("display_text") or ""),
                }
            )

        effective_refs = [
            row
            for row in dedup_candidate_refs
            if str(row.get("memory_id") or "") not in prior_self_ids
            and str(row.get("memory_id") or "") != normalized_memory_id
        ]
        payload = {
            "feedback_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8],
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": str(session_id or ""),
            "query_text": normalized_query,
            "memory_id": normalized_memory_id,
            "feedback_type": normalized_type,
            "lane": normalized_lane,
            "query_self_memory_ids_before": sorted(item for item in prior_self_ids if item),
            "query_self_memory_ids_after": sorted({item for item in prior_self_ids if item} | {normalized_memory_id}),
            "candidate_cache_memory_ids": [str(row.get("memory_id") or "") for row in dedup_candidate_refs],
            "effective_candidate_cache_memory_ids": [str(row.get("memory_id") or "") for row in effective_refs],
            "candidate_refs": dedup_candidate_refs,
            "effective_candidate_refs": effective_refs,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def selected_feedback_for_query(self, *, session_id: str, query_text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in self._load_rows_for_query(session_id=session_id, query_text=query_text):
            memory_id = str(row.get("memory_id") or "")
            feedback_type = str(row.get("feedback_type") or "")
            if memory_id and feedback_type:
                out[memory_id] = feedback_type
        return out

    def _load_rows_for_query(self, *, session_id: str, query_text: str) -> list[dict]:
        if not self._events_path.exists():
            return []
        target_session_id = str(session_id or "")
        target_query = str(query_text or "").strip()
        rows: list[dict] = []
        with self._events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("session_id") or "") != target_session_id:
                    continue
                if str(row.get("query_text") or "").strip() != target_query:
                    continue
                rows.append(row)
        return rows
