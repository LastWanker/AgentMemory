"""编码器接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from ..utils import Vector


class Encoder(ABC):
    """编码器抽象类，负责把文本映射成向量。"""

    def __init__(self, encoder_id: str):
        self.encoder_id = encoder_id

    @abstractmethod
    def encode_tokens(self, text: str) -> Tuple[List[Vector], List[str]]:
        """返回 token 向量组和 token 字符串列表。"""

    @abstractmethod
    def encode_sentence(self, text: str) -> Vector:
        """返回句向量。"""
