"""基于 transformers 的 token-level 编码器。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import os
import time

import torch
from transformers import AutoModel, AutoTokenizer

from .base import Encoder
from .hf_sentence import HFSentenceEncoder
from ..utils import Vector, mean, normalize


class E5TokenEncoder(Encoder):
    """E5 token-level 编码器：输出每个 subword 的向量。"""

    _cuda_logged = False
    _cuda_mem_logged = False

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        use_e5_prefix: bool = False,
        device: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> None:
        super().__init__(encoder_id=f"e5-token@{model_name}")
        timing_log = os.getenv("MEMORY_ENCODER_TIMING", "0") == "1"
        local_files_only = HFSentenceEncoder._resolve_local_files_only(
            model_name, local_files_only
        )
        t0 = time.time()
        if timing_log:
            print(f"[TIMING] T0 start={t0:.2f}")
        model_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        model_elapsed = time.perf_counter() - model_start
        self.model.eval()
        self.device = self._resolve_device(device)
        if self.device.type == "cuda":
            self.model.to(self.device)
        self.use_e5_prefix = use_e5_prefix
        self.model_name = model_name
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if not E5TokenEncoder._cuda_logged:
            E5TokenEncoder._cuda_logged = True
            cuda_available = torch.cuda.is_available()
            print(
                "[CUDA] torch.cuda.is_available()="
                f"{cuda_available} | model_device={self.device} | model_load_s={model_elapsed:.2f}"
            )
            if self.device.type == "cuda":
                print("[设备] E5TokenEncoder 正在使用 CUDA")
                print("CUDA 最大显存（初始化后）:", torch.cuda.max_memory_allocated())
        if timing_log:
            t = time.time()
            _ = self.encode_tokens("hello")
            print(f"[TIMING] T1 first_encode_tokens={time.time() - t:.2f}s")
            print(f"[TIMING] TOTAL={time.time() - t0:.2f}s")

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _maybe_prefix(self, text: str, *, is_query: bool) -> str:
        if self.use_e5_prefix and "e5" in self.model_name.lower():
            return ("query: " if is_query else "passage: ") + text
        return text

    def encode_tokens(self, text: str) -> Tuple[List[Vector], List[str]]:
        # print("[TRACE] 进入 E5TokenEncoder.encode_tokens")
        # print("[TRACE] token_encoder.device =", self.device)
        # print("[TRACE] token_encoder.model param device =", next(self.model.parameters()).device)

        if not text.strip():
            return [], []
        payload = self._maybe_prefix(text, is_query=True)
        inputs = self.tokenizer(
            payload,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if self.device.type == "cuda" and not E5TokenEncoder._cuda_mem_logged:
            E5TokenEncoder._cuda_mem_logged = True
            print("CUDA 最大显存（首次 encode 后）:", torch.cuda.max_memory_allocated())
        hidden = outputs.last_hidden_state[0]
        input_ids = inputs["input_ids"][0].tolist()
        attention = inputs["attention_mask"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        special_ids = set(self.tokenizer.all_special_ids)

        token_vecs: List[Vector] = []
        token_strings: List[str] = []
        for idx, (token_id, mask) in enumerate(zip(input_ids, attention)):
            if mask == 0 or token_id in special_ids:
                continue
            token_vecs.append(hidden[idx].tolist())
            token_strings.append(tokens[idx])

        return token_vecs, token_strings

    def _encode_sentence_with_role(self, text: str, *, is_query: bool) -> Vector:
        payload = self._maybe_prefix(text, is_query=is_query)
        token_vecs, _ = self.encode_tokens(payload)
        if not token_vecs:
            return [0.0] * (self.hidden_size or 1)
        return normalize(mean(token_vecs))

    def encode_query_sentence(self, text: str) -> Vector:
        return self._encode_sentence_with_role(text, is_query=True)

    def encode_passage_sentence(self, text: str) -> Vector:
        return self._encode_sentence_with_role(text, is_query=False)
