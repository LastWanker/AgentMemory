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
    def encode_query_sentence(self, text: str) -> Vector:
        """返回 query 角色的句向量。"""

    @abstractmethod
    def encode_passage_sentence(self, text: str) -> Vector:
        """返回 passage 角色的句向量。"""

    def encode_sentence(self, text: str) -> Vector:
        """已废弃：请显式使用 query/passsage 角色方法。"""

        raise RuntimeError(
            "encode_sentence 已废弃，请 use encode_query_sentence / encode_passage_sentence"
        )
