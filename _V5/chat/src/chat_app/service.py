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
    ) -> ChatResponse:
        real_session_id = session_id or self.new_session()
        history = self._store.load_history(real_session_id)
        retrieval = self._retriever.retrieve_bundle(text, top_k or self._config.top_k)
        prompt_messages = self._build_messages(
            history,
            text,
            coarse_refs=retrieval.coarse_refs,
            association_refs=retrieval.association_refs,
            association_tags=retrieval.association_tags,
        )
        reply = self._client.chat(prompt_messages)
        retrieval_label = retrieval.retrieval_label
        self._store.append_turn(
            real_session_id,
            role="user",
            content=text,
            metadata={"source": "ui", "retrieval_label": retrieval_label},
        )
        self._store.append_turn(
            real_session_id,
            role="assistant",
            content=reply,
            memory_refs=retrieval.coarse_refs,
            metadata={
                "model": self._config.model,
                "retrieval_mode": self._config.retrieval_mode,
                "retrieval_label": retrieval_label,
                "reference_count": len(retrieval.coarse_refs) + len(retrieval.association_refs),
                "coarse_reference_count": len(retrieval.coarse_refs),
                "association_reference_count": len(retrieval.association_refs),
                "association_tags": retrieval.association_tags,
                "association_trace": retrieval.association_trace,
                "prompt_message_count": len(prompt_messages),
            },
        )
        final_history = self._store.load_history(real_session_id)
        return ChatResponse(
            session_id=real_session_id,
            reply=reply,
            memory_refs=retrieval.coarse_refs,
            coarse_memory_refs=retrieval.coarse_refs,
            association_memory_refs=retrieval.association_refs,
            association_tags=retrieval.association_tags,
            association_trace=retrieval.association_trace,
            history=final_history,
            retrieval_label=retrieval_label,
        )

    def _build_messages(
        self,
        history: list[ChatTurn],
        user_text: str,
        *,
        coarse_refs: List[MemoryRef],
        association_refs: List[MemoryRef],
        association_tags: list[dict],
    ) -> list[dict]:
        recent_turns = history[-self._config.history_window :]
        coarse_block = self._format_memory_block(coarse_refs)
        association_block = self._format_association_block(association_refs, association_tags)
        messages = [{"role": "system", "content": self._config.system_prompt}]
        if coarse_block:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是本轮 coarse 粗召回得到的参考记忆。它们更接近直接召回结果；"
                        "只有在相关时才使用，不相关就忽略。\n\n"
                        + coarse_block
                    ),
                }
            )
        if association_block:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是本轮 association 联想召回得到的参考记忆与亮节点标签。"
                        "这些内容来自概念图点亮、上下溯和桥接联想，不代表用户明确提到。"
                        "你可以把它们当作联想线索；只有确实相关时才使用，不相关就忽略。"
                        "如果联想线索和用户问题关系弱，不要硬套。\n\n"
                        + association_block
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
            lines.append(f"[M{idx}] id={ref.memory_id} cluster={ref.cluster_id} text={ref.display_text}")
        return "\n".join(lines)

    @staticmethod
    def _format_association_block(memory_refs: List[MemoryRef], association_tags: list[dict]) -> str:
        lines: list[str] = []
        if association_tags:
            lines.append("联想点亮的较亮节点标签：")
            for idx, tag in enumerate(association_tags[:12], start=1):
                lines.append(
                    f"[T{idx}] level={tag.get('level') or ''} name={tag.get('name') or ''} "
                    f"score={float(tag.get('score') or 0.0):.3f}"
                )
        if memory_refs:
            if lines:
                lines.append("")
            lines.append("联想召回记忆：")
            for idx, ref in enumerate(memory_refs, start=1):
                lines.append(f"[A{idx}] id={ref.memory_id} cluster={ref.cluster_id} text={ref.display_text}")
        return "\n".join(lines)
