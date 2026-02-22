from __future__ import annotations

import re
from typing import Tuple

_WS_RE = re.compile(r"\s+")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{24,}\b")


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = _WS_RE.sub(" ", normalized)
    return normalized


def detect_exclusion_reason(
    text: str,
    *,
    max_chars: int = 4000,
    max_lines: int = 80,
    code_lines_threshold: int = 20,
    max_url_count: int = 15,
) -> str:
    raw = text or ""
    if not raw.strip():
        return "empty"
    if len(raw) >= max_chars:
        return "too_long"

    line_count = raw.count("\n") + 1
    if line_count >= max_lines:
        return "too_many_lines"

    if _URL_RE.findall(raw) and len(_URL_RE.findall(raw)) >= max_url_count:
        return "too_many_urls"

    if len(_HEX_RE.findall(raw)) >= 4:
        return "looks_like_dump"

    if _looks_like_repeated_dump(raw):
        return "repeated_dump"

    if _looks_like_big_code_block(raw, code_lines_threshold=code_lines_threshold):
        return "big_code_block"

    return ""


def _looks_like_big_code_block(text: str, *, code_lines_threshold: int) -> bool:
    fenced = _CODE_FENCE_RE.findall(text)
    if fenced:
        longest = max((block.count("\n") + 1) for block in fenced)
        if longest >= code_lines_threshold:
            return True

    # Fallback heuristic for plain pasted code.
    symbols = "{}();[]=<>"
    symbol_count = sum(text.count(ch) for ch in symbols)
    line_count = text.count("\n") + 1
    if line_count >= code_lines_threshold and symbol_count >= line_count:
        return True
    return False


def _looks_like_repeated_dump(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        return False
    short = [line[:120] for line in lines]
    unique = len(set(short))
    if unique <= max(2, int(0.4 * len(short))):
        return True
    long_lines = sum(1 for line in lines if len(line) >= 180)
    if long_lines >= int(0.6 * len(lines)):
        return True
    return False


def clean_and_flag(text: str) -> Tuple[str, str]:
    reason = detect_exclusion_reason(text)
    return clean_text(text), reason
