from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from agentmemory_v3.config import get_secret
from agentmemory_v3.slots.prompts import IE_SYSTEM_PROMPT, IE_USER_PROMPT_TEMPLATE
from agentmemory_v3.utils.text import tokenize, truncate_text, unique_keep_order


STATUS_KEYWORDS = {
    "完成": "完成",
    "成功": "完成",
    "失败": "失败",
    "报错": "失败",
    "卡住": "卡住",
    "进行中": "进行中",
    "正在": "进行中",
}

EMOTION_KEYWORDS = {
    "焦虑": "焦虑",
    "担心": "焦虑",
    "生气": "生气",
    "开心": "开心",
    "高兴": "开心",
    "烦": "烦躁",
}


@dataclass
class SlotExtractorConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    env_file: str = "data/_secrets/deepseek.env"
    max_tokens: int = 500
    temperature: float = 0.0


class DeepSeekSlotExtractor:
    def __init__(self, config: SlotExtractorConfig) -> None:
        self._config = config
        api_key = get_secret("DEEPSEEK_API_KEY", env_file=config.env_file)
        if not api_key:
            raise RuntimeError(f"DEEPSEEK_API_KEY not found in {config.env_file}")
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def extract(self, *, prev_raw: str, raw: str, next_raw: str) -> dict:
        prompt = IE_USER_PROMPT_TEMPLATE.format(prev_raw=prev_raw or "", raw=raw or "", next_raw=next_raw or "")
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": IE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else "{}"
        payload = json.loads((content or "{}").strip())
        return normalize_slot_payload(payload, prev_raw=prev_raw, raw=raw, next_raw=next_raw, source="deepseek")


def fallback_extract(*, prev_raw: str, raw: str, next_raw: str) -> dict:
    tokens = unique_keep_order([tok for tok in tokenize(raw) if len(tok) >= 2])[:6]
    status = next((label for key, label in STATUS_KEYWORDS.items() if key in raw), "")
    emotion = ""
    for key, label in EMOTION_KEYWORDS.items():
        if key in raw:
            emotion = f"{label}：{truncate_text(raw, 10)}"
            break
    payload = {
        "raw_brief": truncate_text(raw, 24),
        "event": _extract_event(raw),
        "intent": _extract_intent(raw),
        "entities": tokens,
        "status": status,
        "emotion": truncate_text(emotion, 24),
        "context": truncate_text(prev_raw, 24),
        "impact": truncate_text(next_raw, 24),
        "evidence": {
            "raw_brief": truncate_text(raw, 35),
            "event": truncate_text(raw, 35),
            "intent": truncate_text(raw, 35),
            "entities": truncate_text(" ".join(tokens), 35),
            "status": truncate_text(raw, 35),
            "emotion": truncate_text(raw, 35),
            "context": truncate_text(prev_raw, 35),
            "impact": truncate_text(next_raw, 35),
        },
        "warnings": ["fallback_extractor_used"],
        "confidence": 0.3,
    }
    return normalize_slot_payload(payload, prev_raw=prev_raw, raw=raw, next_raw=next_raw, source="fallback")


def normalize_slot_payload(payload: dict, *, prev_raw: str, raw: str, next_raw: str, source: str) -> dict:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    entities = payload.get("entities") if isinstance(payload.get("entities"), list) else []
    entities = [truncate_text(str(item), 12) for item in entities if str(item).strip()][:6]
    out = {
        "raw_brief": truncate_text(str(payload.get("raw_brief") or ""), 24),
        "event": truncate_text(str(payload.get("event") or ""), 24),
        "intent": truncate_text(str(payload.get("intent") or ""), 24),
        "entities": entities,
        "status": truncate_text(str(payload.get("status") or ""), 24),
        "emotion": truncate_text(str(payload.get("emotion") or ""), 24),
        "context": truncate_text(str(payload.get("context") or ""), 24),
        "impact": truncate_text(str(payload.get("impact") or ""), 24),
        "evidence": {
            "raw_brief": truncate_text(str(evidence.get("raw_brief") or ""), 35),
            "event": truncate_text(str(evidence.get("event") or ""), 35),
            "intent": truncate_text(str(evidence.get("intent") or ""), 35),
            "entities": truncate_text(str(evidence.get("entities") or ""), 35),
            "status": truncate_text(str(evidence.get("status") or ""), 35),
            "emotion": truncate_text(str(evidence.get("emotion") or ""), 35),
            "context": truncate_text(str(evidence.get("context") or ""), 35),
            "impact": truncate_text(str(evidence.get("impact") or ""), 35),
        },
        "warnings": list(warnings),
        "confidence": float(payload.get("confidence") or 0.0),
        "slot_source": source,
    }
    if not prev_raw.strip() and not out["context"]:
        out["warnings"].append("context_empty_prev_raw")
    if not next_raw.strip() and not out["impact"]:
        out["warnings"].append("impact_empty_next_raw")
    return out


def _extract_intent(raw: str) -> str:
    patterns = [r"(想要[^，。！？]{1,16})", r"(打算[^，。！？]{1,16})", r"(需要[^，。！？]{1,16})", r"(请[^，。！？]{1,16})"]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return truncate_text(match.group(1), 24)
    return ""


def _extract_event(raw: str) -> str:
    segments = [seg.strip() for seg in re.split(r"[，。！？；]", raw) if seg.strip()]
    return truncate_text(segments[0] if segments else "", 24)
