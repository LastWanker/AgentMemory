from __future__ import annotations

from typing import Any

import numpy as np

from agentmemory_v3.slots.extractor import fallback_extract
from agentmemory_v3.utils.text import tokenize


SLOT_FIELDS = ("raw_brief", "event", "intent", "entities", "status", "emotion", "context", "impact")
SLOT_WEIGHTS = {
    "raw_brief": 0.24,
    "event": 0.20,
    "intent": 0.18,
    "entities": 0.10,
    "status": 0.08,
    "emotion": 0.06,
    "context": 0.07,
    "impact": 0.07,
}
FEATURE_NAMES = (
    "rrf",
    "dense_rank_rrf",
    "bm25_rank_rrf",
    "dense_score",
    "bm25_score",
    "lexical_overlap",
    "entity_overlap",
    "source_dense",
    "source_bm25",
    "source_both",
    "slot_raw_brief",
    "slot_event",
    "slot_intent",
    "slot_entities",
    "slot_status",
    "slot_emotion",
    "slot_context",
    "slot_impact",
    "slot_weighted",
)


def build_query_context(
    dense_index: Any,
    query_text: str,
    slot_row: dict | None = None,
    *,
    q_vec: np.ndarray | None = None,
    query_slot_vectors: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    q_text = (query_text or "").strip()
    q_vec_np = (
        np.asarray(q_vec, dtype=np.float32)
        if q_vec is not None
        else (dense_index.transform_texts([q_text], role="query")[0] if q_text else np.zeros(dense_index.matrix.shape[1], dtype=np.float32))
    )
    query_tokens = tokenize(q_text)
    query_slot_payload = slot_row if isinstance(slot_row, dict) and slot_row else fallback_extract(prev_raw="", raw=q_text, next_raw="")
    if query_slot_vectors is None:
        query_slot_texts = [_slot_text(query_slot_payload.get(field, "")) for field in SLOT_FIELDS]
        query_slot_matrix = dense_index.transform_texts(query_slot_texts, role="query")
        query_slot_vector_map = {field: query_slot_matrix[idx] for idx, field in enumerate(SLOT_FIELDS)}
    else:
        query_slot_vector_map = {field: np.asarray(query_slot_vectors.get(field), dtype=np.float32) for field in SLOT_FIELDS}
    return {
        "query_text": q_text,
        "q_vec": q_vec_np,
        "query_tokens": query_tokens,
        "query_slot_payload": query_slot_payload,
        "query_slot_vectors": query_slot_vector_map,
    }


def build_candidate_feature_map(
    artifacts: Any,
    *,
    query_context: dict[str, Any],
    mem_id: str,
    source_map: dict[str, str],
    coarse_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    idx = artifacts.mem_id_to_idx[mem_id]
    row = artifacts.memory_rows[idx]
    slot_row = artifacts.slot_rows.get(mem_id, {})
    score_map = coarse_scores.get(mem_id, {})
    dense_rank = float(score_map.get("dense_rank", 9999.0))
    bm25_rank = float(score_map.get("bm25_rank", 9999.0))
    dense_score = float(score_map.get("dense", 0.0))
    bm25_score = float(score_map.get("bm25", 0.0))
    rrf = (1.0 / (60.0 + dense_rank)) + (1.0 / (60.0 + bm25_rank))
    query_tokens = query_context["query_tokens"]
    lexical_overlap = _weighted_token_overlap(query_tokens, tokenize(str(row.get("raw_text") or "")))
    entity_overlap = _entity_overlap(query_context["query_slot_payload"], slot_row)
    source = source_map.get(mem_id, "")
    feature_map: dict[str, float] = {
        "rrf": float(rrf),
        "dense_rank_rrf": float(1.0 / (60.0 + dense_rank)),
        "bm25_rank_rrf": float(1.0 / (60.0 + bm25_rank)),
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "lexical_overlap": lexical_overlap,
        "entity_overlap": entity_overlap,
        "source_dense": 1.0 if source == "dense" else 0.0,
        "source_bm25": 1.0 if source == "bm25" else 0.0,
        "source_both": 1.0 if source == "both" else 0.0,
    }
    slot_weighted = 0.0
    for field in SLOT_FIELDS:
        field_matrix = artifacts.slot_vectors.get(field)
        if field_matrix is None or idx >= field_matrix.shape[0]:
            sim = 0.0
        else:
            sim = float(np.dot(field_matrix[idx], query_context["query_slot_vectors"][field]))
        feature_map[f"slot_{field}"] = sim
        slot_weighted += SLOT_WEIGHTS[field] * max(sim, 0.0)
    feature_map["slot_weighted"] = float(slot_weighted)
    return feature_map


def feature_vector_from_map(feature_map: dict[str, float]) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in FEATURE_NAMES]


def heuristic_score(feature_map: dict[str, float]) -> float:
    return (
        0.42 * feature_map["rrf"]
        + 0.26 * feature_map["lexical_overlap"]
        + 0.24 * feature_map["slot_weighted"]
        + 0.08 * feature_map["entity_overlap"]
    )


def slot_text_payload(slot_row: dict) -> dict:
    return {
        "raw_brief": slot_row.get("raw_brief", ""),
        "event": slot_row.get("event", ""),
        "intent": slot_row.get("intent", ""),
        "status": slot_row.get("status", ""),
        "emotion": slot_row.get("emotion", ""),
        "entities": slot_row.get("entities", []),
    }


def _slot_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _entity_overlap(query_slot_payload: dict, slot_row: dict) -> float:
    query_entities = query_slot_payload.get("entities") if isinstance(query_slot_payload, dict) else []
    memory_entities = slot_row.get("entities") if isinstance(slot_row, dict) else []
    if not isinstance(query_entities, list):
        query_entities = []
    if not isinstance(memory_entities, list):
        memory_entities = []
    q_tokens = set(tokenize(" ".join(str(item) for item in query_entities)))
    m_tokens = set(tokenize(" ".join(str(item) for item in memory_entities)))
    if not q_tokens or not m_tokens:
        return 0.0
    return len(q_tokens & m_tokens) / max(1, len(q_tokens))


def _weighted_token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    q_tokens = [token for token in query_tokens if len(token) >= 2]
    if not q_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    matched_weight = sum(len(token) for token in q_tokens if token in doc_set)
    total_weight = sum(len(token) for token in q_tokens)
    return matched_weight / max(1, total_weight)
