"""最小可用编码器：用哈希生成向量。

注意：这不是语义模型，只是为了让框架可运行。
"""

from __future__ import annotations

from typing import List, Tuple

from .base import Encoder
from ..utils import Vector, mean, normalize, stable_hash, tokenize


class SimpleHashEncoder(Encoder):
    """简单哈希编码器，用于本地演示。"""

    def __init__(self, dims: int = 8):
        super().__init__(encoder_id=f"simple-hash@{dims}")
        self.dims = dims

    def encode_tokens(self, text: str) -> Tuple[List[Vector], List[str]]:
        tokens = tokenize(text)
        # 把每个 token 都映射成一个固定维度向量
        vectors = [normalize(stable_hash(token, self.dims)) for token in tokens]
        return vectors, tokens

    def encode_sentence(self, text: str) -> Vector:
        token_vecs, _ = self.encode_tokens(text)
        if not token_vecs:
            # 空文本兜底，返回全零向量
            return [0.0] * self.dims
        sentence_vec = normalize(mean(token_vecs))
        return sentence_vec
