from __future__ import annotations

import re

import numpy as np


FEEDBACK_TYPES = ("unrelated", "toforget")
LANE_TYPES = ("coarse", "association")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def feedback_one_hot(feedback_type: str) -> np.ndarray:
    label = str(feedback_type or "").strip().lower()
    return np.asarray([1.0 if label == item else 0.0 for item in FEEDBACK_TYPES], dtype=np.float32)


def lane_one_hot(lane: str) -> np.ndarray:
    label = str(lane or "").strip().lower()
    return np.asarray([1.0 if label == item else 0.0 for item in LANE_TYPES], dtype=np.float32)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def lexical_overlap_ratio(query_text: str, memory_text: str) -> float:
    q_tokens = set(tokenize(query_text))
    m_tokens = set(tokenize(memory_text))
    if not q_tokens or not m_tokens:
        return 0.0
    return float(len(q_tokens & m_tokens)) / float(max(1, len(q_tokens)))


def explicit_mention_flag(query_text: str, memory_text: str) -> float:
    q = str(query_text or "").strip().lower()
    m = str(memory_text or "").strip().lower()
    if not q or not m:
        return 0.0
    shorter = q if len(q) <= len(m) else m
    longer = m if len(q) <= len(m) else q
    if len(shorter) >= 4 and shorter in longer:
        return 1.0
    return 0.0


def build_feature_vector(
    *,
    query_text: str,
    query_vec: np.ndarray,
    memory_text: str,
    memory_vec: np.ndarray,
    feedback_type: str,
    lane: str = "",
    include_query_vec: bool = True,
    include_memory_vec: bool = True,
    include_cosine: bool = True,
    include_product: bool = True,
    include_abs_diff: bool = True,
    include_feedback_type: bool = True,
    include_lane: bool = True,
    include_lexical_overlap: bool = True,
    include_explicit_mention: bool = True,
) -> np.ndarray:
    q_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    m_vec = np.asarray(memory_vec, dtype=np.float32).reshape(-1)
    if q_vec.shape != m_vec.shape:
        raise ValueError(f"shape mismatch: query_vec={q_vec.shape} memory_vec={m_vec.shape}")

    parts: list[np.ndarray] = []
    if include_feedback_type:
        parts.append(feedback_one_hot(feedback_type))
    if include_lane:
        parts.append(lane_one_hot(lane))
    if include_query_vec:
        parts.append(q_vec)
    if include_memory_vec:
        parts.append(m_vec)
    if include_product:
        parts.append(q_vec * m_vec)
    if include_abs_diff:
        parts.append(np.abs(q_vec - m_vec))
    if include_cosine:
        parts.append(np.asarray([float(np.dot(q_vec, m_vec))], dtype=np.float32))
    if include_lexical_overlap:
        parts.append(np.asarray([lexical_overlap_ratio(query_text, memory_text)], dtype=np.float32))
    if include_explicit_mention:
        parts.append(np.asarray([explicit_mention_flag(query_text, memory_text)], dtype=np.float32))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)
