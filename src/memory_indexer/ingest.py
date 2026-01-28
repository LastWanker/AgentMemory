"""记忆建库入口的轻量预处理：切块与来源标记。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence
import uuid
import re

from .models import MemoryItem


_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


def split_sentences(text: str) -> List[str]:
    """极简句子切分：按中英文句号/问号/感叹号拆分。"""

    return [segment.strip() for segment in _SENTENCE_RE.findall(text) if segment.strip()]


def chunk_text(
    text: str,
    *,
    strategy: str = "sentence_window",
    max_sentences: int = 3,
) -> List[str]:
    """把文本切为多个“记忆块”。"""

    if strategy == "turn":
        # turn 粒度：整段作为一条记忆
        return [text.strip()] if text.strip() else []
    if strategy == "sentence_window":
        sentences = split_sentences(text)
        if not sentences:
            return []
        chunks: List[str] = []
        for i in range(0, len(sentences), max_sentences):
            chunk = "".join(sentences[i : i + max_sentences]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks
    raise ValueError(f"未知切块策略: {strategy}")


def build_memory_items(
    payloads: Iterable[Dict[str, object]],
    *,
    source_default: str = "manual",
    chunk_strategy: str = "sentence_window",
    max_sentences: int = 3,
    tags_default: Sequence[str] | None = None,
) -> List[MemoryItem]:
    """把原始 payload 切块并补全 MemoryItem 元信息。"""

    items: List[MemoryItem] = []
    for payload in payloads:
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        mem_id = str(payload.get("mem_id") or uuid.uuid4())
        source = str(payload.get("source") or source_default)
        tags_value = payload.get("tags") or tags_default or []
        if isinstance(tags_value, str):
            tags = [tags_value]
        else:
            tags = [str(tag) for tag in tags_value]
        base_meta: Dict[str, str] = {
            "chunk_strategy": chunk_strategy,
            "source_hint": source,
        }
        payload_meta = payload.get("meta")
        if isinstance(payload_meta, dict):
            base_meta.update({str(k): str(v) for k, v in payload_meta.items()})

        chunks = chunk_text(text, strategy=chunk_strategy, max_sentences=max_sentences)
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_meta = dict(base_meta)
            chunk_meta["chunk_index"] = str(idx)
            chunk_meta["chunk_total"] = str(total)
            # 单条记忆保持原 mem_id，多条时追加 chunk 标记。
            item_id = mem_id if total <= 1 else f"{mem_id}#c{idx}"
            if total > 1:
                chunk_meta["origin_mem_id"] = mem_id
            items.append(
                MemoryItem(
                    mem_id=item_id,
                    text=chunk,
                    source=source,
                    meta=chunk_meta,
                    summary=payload.get("summary") if isinstance(payload.get("summary"), str) else None,
                    keywords=tags,
                )
            )
    return items
