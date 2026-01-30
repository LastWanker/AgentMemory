"""基于 SentenceTransformer 的句向量编码器。"""

from __future__ import annotations

from typing import List, Tuple

import os
import torch
import time

from sentence_transformers import SentenceTransformer

from .base import Encoder
from ..tokenizers import Tokenizer, TokenizerInput, resolve_tokenizer
from ..utils import Vector


class HFSentenceEncoder(Encoder):
    """句向量编码器：先替换 coarse_vec，让系统更具语义能力。"""

    _cuda_logged = False

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        tokenizer: TokenizerInput = "jieba",
        use_e5_prefix: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(encoder_id=f"hf-sentence@{model_name}")
        timing_log = os.getenv("MEMORY_ENCODER_TIMING", "0") == "1"
        if os.getenv("HF_LOCAL_FILES_ONLY") is not None:
            local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
        t0 = time.time()
        if timing_log:
            print(f"[TIMING] T0 start={t0:.2f}")
        model_start = time.perf_counter()
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        model_elapsed = time.perf_counter() - model_start
        tokenizer_start = time.perf_counter()
        self.tokenizer: Tokenizer = resolve_tokenizer(tokenizer)
        tokenizer_elapsed = time.perf_counter() - tokenizer_start
        self.use_e5_prefix = use_e5_prefix
        self.model_name = model_name
        if not HFSentenceEncoder._cuda_logged:
            HFSentenceEncoder._cuda_logged = True
            cuda_available = torch.cuda.is_available()
            device = getattr(self.model, "device", "unknown")
            print(
                "[CUDA] torch.cuda.is_available()="
                f"{cuda_available} | model_device={device} | "
                f"model_load_s={model_elapsed:.2f} | tokenizer_load_s={tokenizer_elapsed:.2f}"
            )
        if timing_log:
            t = time.time()
            _ = self.tokenizer.tokenize("hello")
            print(f"[TIMING] T1 tokenizer.tokenize={time.time() - t:.2f}s")
            t = time.time()
            _ = self.model.encode("hello", normalize_embeddings=True)
            print(f"[TIMING] T2 first_encode={time.time() - t:.2f}s")
            print(f"[TIMING] TOTAL={time.time() - t0:.2f}s")

    def _maybe_prefix(self, text: str, *, is_query: bool) -> str:
        if self.use_e5_prefix and "e5" in self.model_name.lower():
            return ("query: " if is_query else "passage: ") + text
        return text

    def encode_sentence(self, text: str) -> Vector:
        sent = self._maybe_prefix(text, is_query=False)
        vec = self.model.encode(sent, normalize_embeddings=True).tolist()
        return vec

    def encode_sentence_batch(self, texts: List[str]) -> List[Vector]:
        sentences = [self._maybe_prefix(text, is_query=False) for text in texts]
        vecs = self.model.encode(sentences, normalize_embeddings=True).tolist()
        return vecs

    def encode_query_placeholder_batch(self, texts: List[str]) -> List[Vector]:
        queries = [self._maybe_prefix(text, is_query=True) for text in texts]
        vecs = self.model.encode(queries, normalize_embeddings=True).tolist()
        return vecs

    def encode_tokens(self, text: str) -> Tuple[List[Vector], List[str]]:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return [], []
        placeholder_vec = self.model.encode(
            self._maybe_prefix(text, is_query=True),
            normalize_embeddings=True,
        ).tolist()
        return [placeholder_vec for _ in tokens], tokens
