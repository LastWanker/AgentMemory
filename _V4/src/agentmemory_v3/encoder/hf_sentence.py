from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SentenceEncoderConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    use_e5_prefix: bool = True
    local_files_only: bool = True
    offline: bool = True
    device: str = "auto"
    batch_size: int = 128


class HFSentenceEncoder:
    _MODEL_CACHE: ClassVar[dict[tuple[str, str, bool], SentenceTransformer]] = {}

    def __init__(self, config: SentenceEncoderConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self._configure_runtime()
        cache_key = (config.model_name, str(self.device), bool(config.local_files_only))
        model = self._MODEL_CACHE.get(cache_key)
        if model is None:
            try:
                model = SentenceTransformer(
                    config.model_name,
                    device=str(self.device),
                    local_files_only=bool(config.local_files_only),
                )
            except Exception as exc:
                raise RuntimeError(
                    "HF/E5 encoder initialization failed. "
                    "V4 defaults to local-only + offline. "
                    f"model={config.model_name} device={self.device}"
                ) from exc
            self._MODEL_CACHE[cache_key] = model
        self.model = model
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payloads = [self._maybe_prefix(text, is_query=is_query) for text in texts]
        vectors = self.model.encode(
            payloads,
            batch_size=max(1, int(self.config.batch_size)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query_texts(self, texts: list[str]) -> np.ndarray:
        return self.encode_texts(texts, is_query=True)

    def encode_passage_texts(self, texts: list[str]) -> np.ndarray:
        return self.encode_texts(texts, is_query=False)

    def _maybe_prefix(self, text: str, *, is_query: bool) -> str:
        raw = str(text or "").strip()
        lowered = raw.lower()
        if lowered.startswith("query:") or lowered.startswith("passage:"):
            return raw
        if self.config.use_e5_prefix and "e5" in self.config.model_name.lower():
            return ("query: " if is_query else "passage: ") + raw
        return raw

    def _configure_runtime(self) -> None:
        if self.config.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if self.config.local_files_only:
            os.environ["HF_LOCAL_FILES_ONLY"] = "1"

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        raw = str(device or "").strip().lower()
        if raw in ("", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if raw == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(raw)
