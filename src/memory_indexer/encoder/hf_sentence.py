"""基于 SentenceTransformer 的句向量编码器。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import os
from pathlib import Path
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
        local_files_only: Optional[bool] = None,
    ) -> None:
        super().__init__(encoder_id=f"hf-sentence@{model_name}")
        timing_log = os.getenv("MEMORY_ENCODER_TIMING", "0") == "1"
        local_files_only = self._resolve_local_files_only(model_name, local_files_only)
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
            if cuda_available and str(device).startswith("cuda"):
                print("CUDA 最大显存（句向量初始化后）:", torch.cuda.max_memory_allocated())
        if timing_log:
            t = time.time()
            _ = self.tokenizer.tokenize("hello")
            print(f"[TIMING] T1 tokenizer.tokenize={time.time() - t:.2f}s")
            t = time.time()
            _ = self.model.encode("hello", normalize_embeddings=True)
            print(f"[TIMING] T2 first_encode={time.time() - t:.2f}s")
            print(f"[TIMING] TOTAL={time.time() - t0:.2f}s")

    @staticmethod
    def _resolve_local_files_only(model_name: str, local_files_only: Optional[bool]) -> bool:
        if os.getenv("HF_LOCAL_FILES_ONLY") is not None:
            return os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
        if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
            return True
        if local_files_only is not None:
            return local_files_only
        if os.getenv("HF_HUB_ONLINE") == "1" or os.getenv("TRANSFORMERS_ONLINE") == "1":
            return False
        if os.path.isdir(model_name):
            return True
        cache_dir = HFSentenceEncoder._hf_cache_dir()
        if cache_dir and HFSentenceEncoder._cached_model_exists(cache_dir, model_name):
            return True
        return True

    @staticmethod
    def _hf_cache_dir() -> Optional[Path]:
        env_candidates = (
            os.getenv("HUGGINGFACE_HUB_CACHE"),
            os.getenv("HF_HUB_CACHE"),
            os.getenv("TRANSFORMERS_CACHE"),
        )
        for candidate in env_candidates:
            if candidate:
                return Path(candidate).expanduser()
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        default = Path.home() / ".cache" / "huggingface" / "hub"
        if default.exists():
            return default
        return None

    @staticmethod
    def _cached_model_exists(cache_dir: Path, model_name: str) -> bool:
        if "/" in model_name:
            cache_key = f"models--{model_name.replace('/', '--')}"
            model_dir = cache_dir / cache_key
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
        return False

    def _maybe_prefix(self, text: str, *, is_query: bool) -> str:
        # coarse 角色必须显式区分 query / passage，避免混用导致召回偏移。
        return ("query: " if is_query else "passage: ") + text

    def _encode_with_role(self, text: str, *, is_query: bool) -> Vector:
        payload = self._maybe_prefix(text, is_query=is_query)
        vec = self.model.encode(payload, normalize_embeddings=True).tolist()
        return vec

    def encode_query_sentence(self, text: str) -> Vector:
        """使用 query 前缀编码句向量（L2 归一化）。"""

        return self._encode_with_role(text, is_query=True)

    def encode_passage_sentence(self, text: str) -> Vector:
        """使用 passage 前缀编码句向量（L2 归一化）。"""

        return self._encode_with_role(text, is_query=False)

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
