from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryRef(BaseModel):
    memory_id: str
    cluster_id: str = ""
    score: float = 0.0
    source: str = ""
    display_text: str = ""


class ChatTurn(BaseModel):
    role: str
    content: str
    created_at: str
    memory_refs: List[MemoryRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    memory_refs: List[MemoryRef] = Field(default_factory=list)
    history: List[ChatTurn] = Field(default_factory=list)
    retrieval_label: str = ""


class SessionResponse(BaseModel):
    session_id: str
    history: List[ChatTurn] = Field(default_factory=list)
