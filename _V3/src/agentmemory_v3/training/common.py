from __future__ import annotations

import hashlib
from typing import Iterable

from agentmemory_v3.utils.text import unique_keep_order


def assign_split(query_id: str, *, seed: int = 11, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> str:
    text = f"{seed}:{query_id}".encode("utf-8")
    bucket = int(hashlib.md5(text).hexdigest(), 16) % 10000
    score = bucket / 10000.0
    if score < train_ratio:
        return "train"
    if score < train_ratio + valid_ratio:
        return "valid"
    return "test"


def resolve_query_texts(row: dict) -> list[str]:
    texts: list[str] = []
    base_text = str(row.get("text") or row.get("query_text") or "").strip()
    if base_text:
        texts.append(base_text)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key, value in meta.items():
        lowered = str(key).lower()
        if not any(token in lowered for token in ("augment", "rewrite", "paraphrase", "query", "queries")):
            continue
        texts.extend(_flatten_texts(value))
    return unique_keep_order([text for text in texts if len(text.strip()) >= 2])


def resolve_query_text(row: dict) -> str:
    texts = resolve_query_texts(row)
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    return " ".join(texts)


def _flatten_texts(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        out: list[str] = []
        for inner in value.values():
            out.extend(_flatten_texts(inner))
        return out
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_texts(item))
        return out
    return []
