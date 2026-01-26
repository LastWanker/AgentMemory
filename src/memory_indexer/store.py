"""记忆存储。"""

from __future__ import annotations

from typing import Dict

from .models import EmbeddingRecord, MemoryItem


class MemoryStore:
    """保存 MemoryItem 和 EmbeddingRecord 的简单内存实现。"""

    def __init__(self) -> None:
        self.items: Dict[str, MemoryItem] = {}
        self.embs: Dict[str, EmbeddingRecord] = {}

    def add(self, item: MemoryItem, emb: EmbeddingRecord) -> None:
        self.items[item.mem_id] = item
        self.embs[item.mem_id] = emb
