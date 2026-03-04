from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from agentmemory_v3.utils.text import tokenize, unique_keep_order

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime environment dependent
    OpenAI = None


@dataclass(frozen=True)
class AssociationLLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout_s: float = 60.0
    temperature: float = 0.1
    allow_fallback: bool = True
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    retry_max_backoff_s: float = 8.0


class AssociationLLM:
    def __init__(self, cfg: AssociationLLMConfig) -> None:
        self.cfg = cfg
        self._client = None
        if cfg.api_key and OpenAI is not None:
            self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s)

    def extract_l1_concepts(self, text: str, *, max_items: int) -> list[str]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return []
        payload = self._call_json(
            system_prompt=(
                "你是概念提取器。只输出 JSON。"
                "从原句中提取可用于记忆联想检索的 L1 具体概念。"
                "要求：短、具体、可独立理解；一词多义时加定语；不要大而空。"
            ),
            user_prompt=(
                f"文本：{clean_text}\n"
                f"最多提取 {max(1, int(max_items))} 个概念。\n"
                '返回格式：{"concepts":["..."]}'
            ),
            fallback={"concepts": self._fallback_extract(clean_text, max_items=max_items)},
        )
        concepts = payload.get("concepts", [])
        if not isinstance(concepts, list):
            if not self.cfg.allow_fallback:
                raise RuntimeError("Association LLM payload missing 'concepts' list.")
            return self._fallback_extract(clean_text, max_items=max_items)
        return _clean_concepts(concepts, max_items=max_items)

    def extract_query_concepts(self, query: str, *, max_items: int) -> list[str]:
        clean_query = str(query or "").strip()
        if not clean_query:
            return []
        payload = self._call_json(
            system_prompt=(
                "你是查询概念提取器。只输出 JSON。"
                "把用户 query 提纯成适合联想检索注入的具体概念种子。"
                "优先简短词组，一词多义时加定语。"
            ),
            user_prompt=(
                f"query：{clean_query}\n"
                f"最多 {max(1, int(max_items))} 个概念。\n"
                '返回格式：{"concepts":["..."]}'
            ),
            fallback={"concepts": self._fallback_extract(clean_query, max_items=max_items)},
        )
        concepts = payload.get("concepts", [])
        if not isinstance(concepts, list):
            if not self.cfg.allow_fallback:
                raise RuntimeError("Association LLM payload missing 'concepts' list.")
            return self._fallback_extract(clean_query, max_items=max_items)
        return _clean_concepts(concepts, max_items=max_items)

    def propose_parent_labels(
        self,
        child_names: list[str],
        *,
        target_level: str,
        target_count: int,
    ) -> list[str]:
        clean_children = [item.strip() for item in child_names if str(item).strip()]
        if not clean_children:
            return []
        take = max(1, int(target_count))
        payload = self._call_json(
            system_prompt=(
                "你是概念命名器。只输出 JSON。"
                "你只负责给出父层命名候选，不负责决定具体边。"
                "命名要求：简短、生活化、可读，不要空泛词。"
            ),
            user_prompt=(
                f"目标层级：{target_level}\n"
                f"子概念列表：{json.dumps(clean_children, ensure_ascii=False)}\n"
                f"最多输出 {take} 个命名候选。\n"
                '返回格式：{"labels":["..."]}'
            ),
            fallback={"labels": self._fallback_labels(clean_children, max_items=take)},
        )
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            if not self.cfg.allow_fallback:
                raise RuntimeError("Association LLM payload missing 'labels' list.")
            return self._fallback_labels(clean_children, max_items=take)
        return _clean_concepts(labels, max_items=take)

    def _call_json(self, *, system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            if self.cfg.allow_fallback:
                return dict(fallback)
            raise RuntimeError("Association LLM client unavailable (missing API key or OpenAI SDK).")

        attempts = max(1, int(self.cfg.max_retries))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(self.cfg.temperature),
                    stream=False,
                    response_format={"type": "json_object"},
                )
                text = (response.choices[0].message.content if response.choices else "") or ""
                data = _safe_json_load(text)
                if isinstance(data, dict) and data:
                    return data
                raise ValueError("Empty or invalid JSON payload from LLM.")
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                wait_s = min(
                    float(self.cfg.retry_max_backoff_s),
                    float(self.cfg.retry_backoff_s) * float(2 ** attempt),
                )
                time.sleep(max(0.0, wait_s))
        if not self.cfg.allow_fallback and last_error is not None:
            raise last_error
        return dict(fallback)

    def _fallback_extract(self, text: str, *, max_items: int) -> list[str]:
        tokens = tokenize(text)
        candidates: list[str] = []
        for token in tokens:
            if len(token) <= 1:
                continue
            if token.isdigit():
                continue
            candidates.append(token)
        if not candidates:
            clipped = re.sub(r"\s+", " ", text).strip()
            return [clipped[:20]] if clipped else []
        return unique_keep_order(candidates)[: max(1, int(max_items))]

    def _fallback_labels(self, names: list[str], *, max_items: int) -> list[str]:
        buckets: list[str] = []
        for name in names:
            toks = [tok for tok in tokenize(name) if len(tok) >= 2]
            if toks:
                buckets.append(toks[0])
        if not buckets:
            buckets = [item[:8] for item in names if item]
        return unique_keep_order(buckets)[: max(1, int(max_items))]


def _safe_json_load(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _clean_concepts(values: list[Any], *, max_items: int) -> list[str]:
    out: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip("，,。.;；:：!！?？\"'[]【】()（）")
        if not text:
            continue
        if len(text) > 48:
            text = text[:48].strip()
        out.append(text)
    return unique_keep_order(out)[: max(1, int(max_items))]
