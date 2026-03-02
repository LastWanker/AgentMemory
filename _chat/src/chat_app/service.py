from __future__ import annotations

from typing import List

from .config import ChatAppConfig
from .deepseek_client import DeepSeekChatClient
from .models import ChatResponse, ChatTurn, MemoryRef
from .retriever_adapter import RetrieverAdapter
from .session_store import SessionStore


class ChatService:
    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._store = SessionStore(config.data_dir)
        self._retriever = RetrieverAdapter(config)
        self._client = DeepSeekChatClient(config)

    def new_session(self) -> str:
        return self._store.new_session_id()

    def load_session(self, session_id: str) -> list[ChatTurn]:
        return self._store.load_history(session_id)

    def chat(
        self,
        session_id: str | None,
        text: str,
        top_k: int | None = None,
        coarse_only: bool = False,
    ) -> ChatResponse:
        real_session_id = session_id or self.new_session()
        history = self._store.load_history(real_session_id)
        memory_refs = self._retriever.retrieve(text, top_k or self._config.top_k, coarse_only=coarse_only)
        prompt_messages = self._build_messages(history, text, memory_refs)
        reply = self._client.chat(prompt_messages)
        retrieval_label = "coarse-only" if coarse_only else "reranker"
        self._store.append_turn(
            real_session_id,
            role="user",
            content=text,
            metadata={"source": "ui", "coarse_only": coarse_only, "retrieval_label": retrieval_label},
        )
        self._store.append_turn(
            real_session_id,
            role="assistant",
            content=reply,
            memory_refs=memory_refs,
            metadata={
                "model": self._config.model,
                "retrieval_mode": self._config.retrieval_mode,
                "coarse_only": coarse_only,
                "retrieval_label": retrieval_label,
                "reference_count": len(memory_refs),
                "prompt_message_count": len(prompt_messages),
            },
        )
        final_history = self._store.load_history(real_session_id)
        return ChatResponse(
            session_id=real_session_id,
            reply=reply,
            memory_refs=memory_refs,
            history=final_history,
            coarse_only=coarse_only,
            retrieval_label=retrieval_label,
        )

    def _build_messages(self, history: list[ChatTurn], user_text: str, memory_refs: List[MemoryRef]) -> list[dict]:
        recent_turns = history[-self._config.history_window :]
        memory_block = self._format_memory_block(memory_refs)
        messages = [{"role": "system", "content": self._config.system_prompt}]
        if memory_block:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是本轮检索得到的参考记忆。只有在相关时才使用，不相关就忽略。\n\n" + memory_block
                    ),
                }
            )
        for turn in recent_turns:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _format_memory_block(memory_refs: List[MemoryRef]) -> str:
        if not memory_refs:
            return ""
        lines: list[str] = []
        for idx, ref in enumerate(memory_refs, start=1):
            slot_summary = []
            for key in ("raw_brief", "event", "intent", "status", "emotion"):
                value = ref.slot_texts.get(key) if isinstance(ref.slot_texts, dict) else None
                if value:
                    slot_summary.append(f"{key}={value}")
            suffix = f" | {'; '.join(slot_summary)}" if slot_summary else ""
            lines.append(f"[M{idx}] id={ref.memory_id} cluster={ref.cluster_id} text={ref.display_text}{suffix}")
        return "\n".join(lines)
