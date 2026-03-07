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
    _FORCE_OFFLINE_WARNED: ClassVar[bool] = False

    def __init__(self, config: SentenceEncoderConfig) -> None:
        # V5 hard requirement: E5 must run local-only + offline to avoid hidden HF network waits.
        self.config = SentenceEncoderConfig(
            model_name=str(config.model_name),
            use_e5_prefix=bool(config.use_e5_prefix),
            local_files_only=True,
            offline=True,
            device=str(config.device),
            batch_size=int(config.batch_size),
        )
        if (not bool(config.local_files_only) or not bool(config.offline)) and not self._FORCE_OFFLINE_WARNED:
            self._FORCE_OFFLINE_WARNED = True
            print("[v5][encoder] forcing HF local-only + offline for E5 runtime")
        self.device = self._resolve_device(self.config.device)
        self._configure_runtime()
        model_ref = self._resolve_model_ref(
            self.config.model_name,
            local_files_only=True,
            offline=True,
        )
        cache_key = (str(model_ref), str(self.device), True)
        model = self._MODEL_CACHE.get(cache_key)
        if model is None:
            try:
                try:
                    model = SentenceTransformer(
                        model_ref,
                        device=str(self.device),
                        local_files_only=True,
                    )
                except TypeError:
                    # Older sentence-transformers releases do not expose local_files_only.
                    # Offline behavior is still enforced through HF_* env vars above.
                    model = SentenceTransformer(
                        model_ref,
                        device=str(self.device),
                    )
            except Exception as exc:
                raise RuntimeError(
                    "HF/E5 encoder initialization failed. "
                    "V5 defaults to local-only + offline. "
                    f"model={self.config.model_name} resolved={model_ref} device={self.device}"
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
        os.environ["HF_LOCAL_FILES_ONLY"] = "1" if self.config.local_files_only else "0"
        os.environ["HF_HUB_OFFLINE"] = "1" if self.config.offline else "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "1" if self.config.offline else "0"

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        raw = str(device or "").strip().lower()
        if raw in ("", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if raw == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(raw)

    @staticmethod
    def _resolve_model_ref(model_name: str, *, local_files_only: bool, offline: bool) -> str:
        text = str(model_name or "").strip()
        if not text:
            return text
        path = Path(text)
        if path.exists():
            return str(path)
        if not (local_files_only or offline):
            return text
        cached = HFSentenceEncoder._find_cached_snapshot(text)
        if cached:
            return cached
        raise RuntimeError(
            "HF/E5 local model not found in cache while local-only/offline is enabled. "
            f"model={text} hf_cache_dirs={HFSentenceEncoder._hf_cache_dirs()}"
        )

    @staticmethod
    def _hf_cache_dirs() -> list[str]:
        candidates: list[Path] = []
        for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
            value = str(os.getenv(key, "")).strip()
            if value:
                candidates.append(Path(value).expanduser())
        hf_home = str(os.getenv("HF_HOME", "")).strip()
        if hf_home:
            candidates.append(Path(hf_home).expanduser() / "hub")
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = str(item)
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    @staticmethod
    def _find_cached_snapshot(model_name: str) -> str:
        if "/" not in model_name:
            return ""
        repo_dir_name = f"models--{model_name.replace('/', '--')}"
        for cache_dir in HFSentenceEncoder._hf_cache_dirs():
            base = Path(cache_dir)
            snapshots = base / repo_dir_name / "snapshots"
            if not snapshots.exists():
                continue
            folders = [item for item in snapshots.iterdir() if item.is_dir()]
            if not folders:
                continue
            folders.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            return str(folders[0])
        return ""
