from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .models import QuerySample


def build_query_samples(
    memories: Sequence[Dict[str, object]],
    *,
    supplemental_path: Path | None = None,
) -> List[QuerySample]:
    """Build query samples from memory rows.

    Base channel is always identity queries (memory text -> query text).
    Supplemental queries are optional and appended from an external JSONL file.
    """

    base_samples = _build_identity_samples(memories)
    if supplemental_path is None:
        return base_samples
    supplemental = load_supplemental_query_samples(supplemental_path)
    return merge_query_samples(base_samples, supplemental)


def load_supplemental_query_samples(path: Path) -> List[QuerySample]:
    """Load optional supplemental query rows from an external JSONL file."""

    if not path.exists():
        return []
    samples: List[QuerySample] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        query_text = str(payload.get("query_text", "")).strip()
        if not query_text:
            continue
        positives = _normalize_list(payload.get("positives"))
        mem_id = str(payload.get("mem_id", "")).strip()
        if not positives and mem_id:
            positives = [mem_id]
        if not positives:
            continue
        candidates = _normalize_list(payload.get("candidates")) or list(positives)
        hard_negatives = _normalize_list(payload.get("hard_negatives"))
        query_id = str(payload.get("query_id", "")).strip() or f"supq-{idx:06d}"
        meta = payload.get("meta")
        samples.append(
            QuerySample(
                query_id=query_id,
                query_text=query_text,
                positives=positives,
                candidates=candidates,
                hard_negatives=hard_negatives,
                meta=meta if isinstance(meta, dict) else {"source": "supplemental_query_file"},
            )
        )
    return samples


def merge_query_samples(base: Sequence[QuerySample], supplemental: Sequence[QuerySample]) -> List[QuerySample]:
    """Append supplemental queries without replacing base identity queries."""

    merged: List[QuerySample] = list(base)
    seen_ids = {row.query_id for row in merged}
    seq = 1
    for row in supplemental:
        query_id = row.query_id
        if not query_id or query_id in seen_ids:
            while True:
                candidate = f"supq-{seq:06d}"
                seq += 1
                if candidate not in seen_ids:
                    query_id = candidate
                    break
        seen_ids.add(query_id)
        merged.append(
            QuerySample(
                query_id=query_id,
                query_text=row.query_text,
                positives=list(row.positives),
                candidates=list(row.candidates),
                hard_negatives=list(row.hard_negatives),
                meta=dict(row.meta),
            )
        )
    return merged


def to_followup_rows(samples: Sequence[QuerySample]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in samples:
        out.append(
            {
                "query_id": item.query_id,
                "query_text": item.query_text,
                "positives": item.positives,
                "candidates": item.candidates,
                "hard_negatives": item.hard_negatives,
                "meta": item.meta,
            }
        )
    return out


def to_eval_rows(samples: Sequence[QuerySample]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in samples:
        out.append(
            {
                "query_id": item.query_id,
                "query_text": item.query_text,
                "expected_mem_ids": item.positives,
                "meta": item.meta,
            }
        )
    return out


def _build_identity_samples(memories: Sequence[Dict[str, object]]) -> List[QuerySample]:
    rows: List[QuerySample] = []
    for idx, row in enumerate(memories, start=1):
        mem_id = str(row.get("mem_id", "")).strip()
        query_text = str(row.get("text", "")).strip()
        if not mem_id or not query_text:
            continue
        cluster_id = str(row.get("cluster_id", "")).strip()
        rows.append(
            QuerySample(
                query_id=f"chatq-{idx:06d}",
                query_text=query_text,
                positives=[mem_id],
                # Identity channel keeps candidates minimal; enrichment should be external.
                candidates=[mem_id],
                hard_negatives=[],
                meta={
                    "source": "memory_identity",
                    "channel": "identity",
                    "mem_id": mem_id,
                    "cluster_id": cluster_id,
                    "note": "supplemental_queries_should_be_appended_externally",
                },
            )
        )
    return rows


def _normalize_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out
