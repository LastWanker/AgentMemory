from __future__ import annotations

import re
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    base_tokens = TOKEN_RE.findall((text or "").lower())
    out: list[str] = []
    for token in base_tokens:
        if not token:
            continue
        out.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            n = len(token)
            if n <= 8:
                out.append(token)
            for size in (2, 3):
                if n < size:
                    continue
                for idx in range(0, n - size + 1):
                    out.append(token[idx : idx + size])
    return out


def truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text[:limit] if len(text) > limit else text


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
