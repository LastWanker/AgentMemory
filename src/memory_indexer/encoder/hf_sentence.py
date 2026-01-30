"""基于 SentenceTransformer 的句向量编码器。"""

from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from .base import Encoder
from ..tokenizers import Tokenizer, TokenizerInput, resolve_tokenizer
from ..utils import Vector


class HFSentenceEncoder(Encoder):
    """句向量编码器：先替换 coarse_vec，让系统更具语义能力。"""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        tokenizer: TokenizerInput = "jieba",
        use_e5_prefix: bool = True,
    ) -> None:
        super().__init__(encoder_id=f"hf-sentence@{model_name}")
        self.model = SentenceTransformer(model_name)
        self.tokenizer: Tokenizer = resolve_tokenizer(tokenizer)
        self.use_e5_prefix = use_e5_prefix
        self.model_name = model_name

    def _maybe_prefix(self, text: str, *, is_query: bool) -> str:
        if self.use_e5_prefix and "e5" in self.model_name.lower():
            return ("query: " if is_query else "passage: ") + text
        return text

    def encode_sentence(self, text: str) -> Vector:
        sent = self._maybe_prefix(text, is_query=False)
        vec = self.model.encode(sent, normalize_embeddings=True).tolist()
        return vec

    def encode_tokens(self, text: str) -> Tuple[List[Vector], List[str]]:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return [], []
        placeholder_vec = self.model.encode(
            self._maybe_prefix(text, is_query=True),
            normalize_embeddings=True,
        ).tolist()
        return [placeholder_vec for _ in tokens], tokens
