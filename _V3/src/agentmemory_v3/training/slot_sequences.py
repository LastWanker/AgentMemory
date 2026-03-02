from __future__ import annotations

from pathlib import Path

import numpy as np

from agentmemory_v3.slots.extractor import fallback_extract
from agentmemory_v3.retrieval.feature_builder import SLOT_FIELDS
from agentmemory_v3.training.common import resolve_query_text
from agentmemory_v3.utils.io import read_jsonl


def memory_slot_sequences(slot_vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(slot_vectors[field], dtype=np.float32) for field in SLOT_FIELDS], axis=1)
    mask = np.linalg.norm(stacked, axis=-1) > 1e-8
    empty_rows = ~mask.any(axis=1)
    if np.any(empty_rows):
        mask[empty_rows, 0] = True
    return stacked.astype(np.float32), mask.astype(bool)


def load_query_slot_sequences(query_slots_path: Path, dense_index, *, query_path: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    slot_map = {str(row.get("query_id") or ""): row for row in read_jsonl(query_slots_path)} if query_slots_path.exists() else {}
    if query_path is None:
        rows = list(slot_map.values())
    else:
        rows = []
        for query_row in read_jsonl(query_path):
            query_id = str(query_row.get("query_id") or "")
            slot_row = slot_map.get(query_id)
            if slot_row is None:
                text = resolve_query_text(query_row)
                slot_row = {"query_id": query_id, "text": text, **fallback_extract(prev_raw="", raw=text, next_raw="")}
            rows.append(slot_row)
    if not rows:
        return {}
    field_texts = {field: [] for field in SLOT_FIELDS}
    query_ids = []
    for row in rows:
        query_ids.append(str(row.get("query_id") or ""))
        for field in SLOT_FIELDS:
            value = row.get(field, "")
            if isinstance(value, list):
                value = " ".join(str(item) for item in value if str(item).strip())
            field_texts[field].append(str(value or ""))
    field_vectors = {field: dense_index.transform_texts(texts) for field, texts in field_texts.items()}
    out = {}
    for row_idx, query_id in enumerate(query_ids):
        seq = np.stack([field_vectors[field][row_idx] for field in SLOT_FIELDS], axis=0).astype(np.float32)
        mask = (np.linalg.norm(seq, axis=-1) > 1e-8).astype(bool)
        if not mask.any():
            mask[0] = True
        out[query_id] = (seq, mask)
    return out


def ensure_query_slot_row_map(query_path: Path, query_slots_path: Path) -> dict[str, dict]:
    slot_map = {str(row.get("query_id") or ""): row for row in read_jsonl(query_slots_path)} if query_slots_path.exists() else {}
    if slot_map:
        return slot_map
    out = {}
    for row in read_jsonl(query_path):
        out[str(row.get("query_id") or "")] = {"query_id": str(row.get("query_id") or ""), "text": resolve_query_text(row)}
    return out
