from __future__ import annotations

from typing import List

from openai import OpenAI

from .config import ChatAppConfig


class DeepSeekChatClient:
    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(self, messages: List[dict]) -> str:
        if not self._config.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=0.4,
            stream=False,
        )
        message = response.choices[0].message.content if response.choices else ""
        return (message or "").strip()
