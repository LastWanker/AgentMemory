from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import List
from uuid import uuid4

from .config import ChatAppConfig
from .deepseek_client import DeepSeekChatClient
from .feedback_store import FeedbackStore
from .models import ChatResponse, ChatRetrieveResponse, ChatTurn, MemoryRef
from .retriever_adapter import RetrievalBundle, RetrieverAdapter
from .session_store import SessionStore


class ChatService:
    def __init__(self, config: ChatAppConfig) -> None:
        self._config = config
        self._store = SessionStore(config.data_dir)
        self._feedback = FeedbackStore(config.feedback_dir)
        self._retriever = RetrieverAdapter(config)
        self._client = DeepSeekChatClient(config)
        self._pending_retrievals: dict[str, dict] = {}
        self._pending_lock = Lock()
        self._pending_ttl = timedelta(minutes=15)

    def new_session(self) -> str:
        return self._store.new_session_id()

    def load_session(self, session_id: str) -> list[ChatTurn]:
        return self._store.load_history(session_id)

    def chat(
        self,
        session_id: str | None,
        text: str,
        top_k: int | None = None,
        memory_preference_enabled: bool | None = None,
    ) -> ChatResponse:
        retrieval = self.chat_retrieve(session_id, text, top_k, memory_preference_enabled)
        return self.chat_respond(retrieval.session_id, retrieval.retrieval_id)

    def chat_retrieve(
        self,
        session_id: str | None,
        text: str,
        top_k: int | None = None,
        memory_preference_enabled: bool | None = None,
    ) -> ChatRetrieveResponse:
        real_session_id = session_id or self.new_session()
        history = self._store.load_history(real_session_id)
        retrieval = self._retriever.retrieve_bundle(
            text,
            top_k or self._config.top_k,
            memory_preference_enabled=memory_preference_enabled,
        )
        prompt_messages = self._build_messages(
            history,
            text,
            coarse_refs=retrieval.coarse_refs,
            association_refs=retrieval.association_refs,
            association_tags=retrieval.association_tags,
        )
        retrieval_label = retrieval.retrieval_label
        retrieval_id = self._put_pending_retrieval(
            session_id=real_session_id,
            text=text,
            retrieval=retrieval,
            prompt_messages=prompt_messages,
        )
        self._store.append_turn(
            real_session_id,
            role="user",
            content=text,
            metadata={
                "source": "ui",
                "retrieval_label": retrieval_label,
                "memory_preference_enabled": (
                    bool(memory_preference_enabled) if memory_preference_enabled is not None else None
                ),
            },
        )
        self._store.append_turn(
            real_session_id,
            role="assistant",
            content="记忆已召回",
            memory_refs=retrieval.coarse_refs,
            metadata={
                "stage": "retrieval_ready",
                "retrieval_label": retrieval_label,
                "reference_count": len(retrieval.coarse_refs) + len(retrieval.association_refs),
                "coarse_reference_count": len(retrieval.coarse_refs),
                "association_reference_count": len(retrieval.association_refs),
                "association_tags": retrieval.association_tags,
                "association_trace": retrieval.association_trace,
                "coarse_memory_refs": [ref.model_dump() for ref in retrieval.coarse_refs],
                "association_memory_refs": [ref.model_dump() for ref in retrieval.association_refs],
                "suppressor_trace": retrieval.suppressor_trace,
                "retrieval_id": retrieval_id,
                "memory_preference_enabled": (
                    bool(memory_preference_enabled) if memory_preference_enabled is not None else None
                ),
            },
        )
        final_history = self._store.load_history(real_session_id)
        return ChatRetrieveResponse(
            session_id=real_session_id,
            retrieval_id=retrieval_id,
            reply="记忆已召回",
            memory_refs=retrieval.coarse_refs,
            coarse_memory_refs=retrieval.coarse_refs,
            association_memory_refs=retrieval.association_refs,
            association_tags=retrieval.association_tags,
            association_trace=retrieval.association_trace,
            suppressor_trace=retrieval.suppressor_trace,
            history=final_history,
            retrieval_label=retrieval_label,
        )

    def chat_respond(self, session_id: str, retrieval_id: str) -> ChatResponse:
        pending = self._get_pending_retrieval(retrieval_id)
        if pending is None:
            raise KeyError("retrieval_id not found or expired")
        if str(pending["session_id"]) != str(session_id):
            raise KeyError("retrieval_id does not belong to this session")
        prompt_messages = list(pending["prompt_messages"])
        retrieval = pending["retrieval"]
        reply = self._client.chat(prompt_messages)
        self._store.append_turn(
            session_id,
            role="assistant",
            content=reply,
            memory_refs=retrieval.coarse_refs,
            metadata={
                "stage": "llm_reply",
                "model": self._config.model,
                "retrieval_mode": self._config.retrieval_mode,
                "retrieval_label": retrieval.retrieval_label,
                "reference_count": len(retrieval.coarse_refs) + len(retrieval.association_refs),
                "coarse_reference_count": len(retrieval.coarse_refs),
                "association_reference_count": len(retrieval.association_refs),
                "association_tags": retrieval.association_tags,
                "association_trace": retrieval.association_trace,
                "coarse_memory_refs": [ref.model_dump() for ref in retrieval.coarse_refs],
                "association_memory_refs": [ref.model_dump() for ref in retrieval.association_refs],
                "suppressor_trace": retrieval.suppressor_trace,
                "prompt_message_count": len(prompt_messages),
                "retrieval_id": retrieval_id,
            },
        )
        self._pop_pending_retrieval(retrieval_id)
        final_history = self._store.load_history(session_id)
        return ChatResponse(
            session_id=session_id,
            retrieval_id=retrieval_id,
            reply=reply,
            memory_refs=retrieval.coarse_refs,
            coarse_memory_refs=retrieval.coarse_refs,
            association_memory_refs=retrieval.association_refs,
            association_tags=retrieval.association_tags,
            association_trace=retrieval.association_trace,
            suppressor_trace=retrieval.suppressor_trace,
            history=final_history,
            retrieval_label=retrieval.retrieval_label,
        )

    def record_feedback(
        self,
        *,
        session_id: str,
        query_text: str,
        memory_id: str,
        feedback_type: str,
        lane: str,
        candidate_refs: list[dict],
    ) -> dict:
        row = self._feedback.append_feedback(
            session_id=session_id,
            query_text=query_text,
            memory_id=memory_id,
            feedback_type=feedback_type,
            lane=lane,
            candidate_refs=candidate_refs,
        )
        selected = self._feedback.selected_feedback_for_query(session_id=session_id, query_text=query_text)
        return {
            "ok": True,
            "feedback_id": str(row.get("feedback_id") or ""),
            "selected_feedback": selected,
            "stored_row": row,
        }

    def _put_pending_retrieval(
        self,
        *,
        session_id: str,
        text: str,
        retrieval: RetrievalBundle,
        prompt_messages: list[dict],
    ) -> str:
        retrieval_id = uuid4().hex
        now = datetime.now(timezone.utc)
        payload = {
            "session_id": session_id,
            "text": text,
            "retrieval": retrieval,
            "prompt_messages": prompt_messages,
            "created_at": now,
        }
        with self._pending_lock:
            self._purge_expired_pending_locked(now)
            self._pending_retrievals[retrieval_id] = payload
        return retrieval_id

    def _get_pending_retrieval(self, retrieval_id: str) -> dict | None:
        with self._pending_lock:
            self._purge_expired_pending_locked(datetime.now(timezone.utc))
            payload = self._pending_retrievals.get(retrieval_id)
            if payload is None:
                return None
            return dict(payload)

    def _pop_pending_retrieval(self, retrieval_id: str) -> None:
        with self._pending_lock:
            self._pending_retrievals.pop(retrieval_id, None)

    def _purge_expired_pending_locked(self, now: datetime) -> None:
        expired = [
            retrieval_id
            for retrieval_id, payload in self._pending_retrievals.items()
            if now - payload.get("created_at", now) > self._pending_ttl
        ]
        for retrieval_id in expired:
            self._pending_retrievals.pop(retrieval_id, None)

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
