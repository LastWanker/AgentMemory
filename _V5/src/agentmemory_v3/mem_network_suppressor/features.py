from __future__ import annotations

import numpy as np

FEEDBACK_TYPE_TO_ID = {
    "unrelated": 0,
    # Single-head mode: toforget is trained with the same suppress semantics as unrelated.
    "toforget": 0,
}
UNKNOWN_FEEDBACK_TYPE_ID = 1
FEEDBACK_TYPE_VOCAB_SIZE = 2

LANE_TO_ID = {
    "coarse": 0,
    "association": 1,
}
UNKNOWN_LANE_ID = 2
LANE_VOCAB_SIZE = 3
UNKNOWN_MEMORY_ID_INDEX = 0


def _softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    x = x - float(np.max(x))
    exp = np.exp(x, dtype=np.float32)
    denom = float(np.sum(exp))
    if denom <= 1e-8:
        return np.full_like(x, 1.0 / float(max(1, x.size)))
    return exp / denom


def build_feature_vector(
    *,
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feedback_agg_vec: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    f = np.asarray(feedback_agg_vec, dtype=np.float32).reshape(-1)
    if not (q.shape == c.shape == f.shape):
        raise ValueError(f"shape mismatch: query={q.shape}, candidate={c.shape}, feedback_agg={f.shape}")
    return np.concatenate([q, c, f], axis=0).astype(np.float32, copy=False)


def feedback_type_to_id(value: str) -> int:
    key = str(value or "").strip().lower()
    return int(FEEDBACK_TYPE_TO_ID.get(key, UNKNOWN_FEEDBACK_TYPE_ID))


def lane_to_id(value: str) -> int:
    key = str(value or "").strip().lower()
    return int(LANE_TO_ID.get(key, UNKNOWN_LANE_ID))


def build_memory_id_lookup(rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        memory_id = str(row.get("m_id") or row.get("memory_id") or "").strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        ordered_ids.append(memory_id)
    # Index 0 is reserved for unknown memory ids.
    index_by_id = {memory_id: idx + 1 for idx, memory_id in enumerate(ordered_ids)}
    return ordered_ids, index_by_id


def memory_id_to_index(memory_id: str, memory_id_to_index_map: dict[str, int]) -> int:
    key = str(memory_id or "").strip()
    if not key:
        return int(UNKNOWN_MEMORY_ID_INDEX)
    return int(memory_id_to_index_map.get(key, UNKNOWN_MEMORY_ID_INDEX))


def build_feedback_type_lane_arrays(feedback_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    size = int(len(feedback_rows))
    type_ids = np.full((size,), UNKNOWN_FEEDBACK_TYPE_ID, dtype=np.int64)
    lane_ids = np.full((size,), UNKNOWN_LANE_ID, dtype=np.int64)
    for idx, row in enumerate(feedback_rows):
        type_ids[idx] = feedback_type_to_id(str(row.get("feedback_type") or ""))
        lane_ids[idx] = lane_to_id(str(row.get("lane") or ""))
    return type_ids, lane_ids


def select_feedback_top_indices(
    *,
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feedback_query_matrix: np.ndarray,
    feedback_memory_matrix: np.ndarray,
    feedback_memory_id_ids: np.ndarray | None,
    candidate_memory_id_id: int,
    top_k: int,
    query_weight: float = 0.7,
    candidate_weight: float = 0.3,
    memory_id_match_bonus: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    fq = np.asarray(feedback_query_matrix, dtype=np.float32)
    fm = np.asarray(feedback_memory_matrix, dtype=np.float32)
    if fq.size == 0 or fm.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    if fq.shape != fm.shape or fq.shape[1] != q.size or c.size != q.size:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    scores = float(query_weight) * (fq @ q) + float(candidate_weight) * (fm @ c)
    if (
        feedback_memory_id_ids is not None
        and int(candidate_memory_id_id) > int(UNKNOWN_MEMORY_ID_INDEX)
        and int(np.asarray(feedback_memory_id_ids).shape[0]) == int(scores.shape[0])
    ):
        ids = np.asarray(feedback_memory_id_ids, dtype=np.int64).reshape(-1)
        match = (ids == int(candidate_memory_id_id)).astype(np.float32)
        if np.any(match > 0.0):
            scores = scores + float(memory_id_match_bonus) * match
    k = min(max(1, int(top_k)), int(scores.shape[0]))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    top_scores = scores[top_idx].astype(np.float32, copy=False)
    return top_idx.astype(np.int64, copy=False), top_scores


def build_feedback_memory_slots(
    *,
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feedback_query_matrix: np.ndarray,
    feedback_memory_matrix: np.ndarray,
    feedback_type_ids: np.ndarray,
    feedback_lane_ids: np.ndarray,
    feedback_memory_id_ids: np.ndarray | None,
    candidate_memory_id_id: int,
    top_k: int,
    memory_id_match_bonus: float = 0.5,
) -> dict:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    dim = int(q.size)
    k = max(1, int(top_k))
    slot_query = np.zeros((k, dim), dtype=np.float32)
    slot_memory = np.zeros((k, dim), dtype=np.float32)
    slot_type_ids = np.full((k,), UNKNOWN_FEEDBACK_TYPE_ID, dtype=np.int64)
    slot_lane_ids = np.full((k,), UNKNOWN_LANE_ID, dtype=np.int64)
    slot_memory_id_ids = np.full((k,), UNKNOWN_MEMORY_ID_INDEX, dtype=np.int64)
    slot_mask = np.zeros((k,), dtype=np.float32)
    top_idx, top_scores = select_feedback_top_indices(
        query_vec=q,
        candidate_vec=candidate_vec,
        feedback_query_matrix=feedback_query_matrix,
        feedback_memory_matrix=feedback_memory_matrix,
        feedback_memory_id_ids=feedback_memory_id_ids,
        candidate_memory_id_id=int(candidate_memory_id_id),
        top_k=k,
        memory_id_match_bonus=float(memory_id_match_bonus),
    )
    valid = min(int(k), int(top_idx.shape[0]))
    if valid > 0:
        chosen = top_idx[:valid]
        slot_query[:valid] = np.asarray(feedback_query_matrix[chosen], dtype=np.float32)
        slot_memory[:valid] = np.asarray(feedback_memory_matrix[chosen], dtype=np.float32)
        slot_type_ids[:valid] = np.asarray(feedback_type_ids[chosen], dtype=np.int64)
        slot_lane_ids[:valid] = np.asarray(feedback_lane_ids[chosen], dtype=np.int64)
        if feedback_memory_id_ids is not None:
            slot_memory_id_ids[:valid] = np.asarray(np.asarray(feedback_memory_id_ids)[chosen], dtype=np.int64)
        slot_mask[:valid] = 1.0
    return {
        "slot_query": slot_query,
        "slot_memory": slot_memory,
        "slot_type_ids": slot_type_ids,
        "slot_lane_ids": slot_lane_ids,
        "slot_memory_id_ids": slot_memory_id_ids,
        "slot_mask": slot_mask,
        "candidate_memory_id_id": int(candidate_memory_id_id),
        "top_indices": [int(i) for i in top_idx[:valid].tolist()],
        "top_weights": [float(v) for v in _softmax(top_scores[:valid]).tolist()] if valid > 0 else [],
    }


def aggregate_feedback_context(
    *,
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feedback_query_matrix: np.ndarray,
    feedback_memory_matrix: np.ndarray,
    top_k: int,
    query_weight: float = 0.7,
    candidate_weight: float = 0.3,
) -> tuple[np.ndarray, dict]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    fq = np.asarray(feedback_query_matrix, dtype=np.float32)
    fm = np.asarray(feedback_memory_matrix, dtype=np.float32)
    if fq.size == 0 or fm.size == 0:
        dim = int(q.size) if q.size > 0 else int(c.size)
        return np.zeros((dim,), dtype=np.float32), {"top_indices": [], "top_weights": []}
    if fq.shape != fm.shape or fq.shape[1] != q.size or c.size != q.size:
        dim = int(q.size)
        return np.zeros((dim,), dtype=np.float32), {"top_indices": [], "top_weights": []}

    s_query = fq @ q
    s_candidate = fm @ c
    scores = float(query_weight) * s_query + float(candidate_weight) * s_candidate
    k = min(max(1, int(top_k)), int(scores.shape[0]))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    top_scores = scores[top_idx]
    weights = _softmax(top_scores)
    agg = np.sum(fm[top_idx] * weights[:, None], axis=0, dtype=np.float32)
    return agg.astype(np.float32, copy=False), {
        "top_indices": [int(item) for item in top_idx.tolist()],
        "top_weights": [float(item) for item in weights.tolist()],
    }
