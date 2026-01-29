"""记忆存储。"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import EmbeddingRecord, MemoryItem


class MemoryStore:
    """保存 MemoryItem 和 EmbeddingRecord 的简单内存实现。"""

    def __init__(self) -> None:
        self.items: Dict[str, MemoryItem] = {}
        self.embs: Dict[str, EmbeddingRecord] = {}
        # 额外保存 token 列表，为词法证据与可解释性提供支持。
        self.tokens: Dict[str, List[str]] = {}

    def add(self, item: MemoryItem, emb: EmbeddingRecord, tokens: Optional[List[str]] = None) -> None:
        self.items[item.mem_id] = item
        self.embs[item.mem_id] = emb
        if tokens is not None:
            self.tokens[item.mem_id] = tokens
