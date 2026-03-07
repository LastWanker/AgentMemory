from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryRef(BaseModel):
    memory_id: str
    cluster_id: str = ""
    score: float = 0.0
    base_score: float = 0.0
    source: str = ""
    display_text: str = ""
    suppressed: bool = False
    suppress_score: float = 0.0
    suppress_delta: float = 0.0
    suppress_reason: str = ""
    suppress_lane: str = ""


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
    memory_preference_enabled: Optional[bool] = None


class ChatRetrieveRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    top_k: Optional[int] = None
    memory_preference_enabled: Optional[bool] = None


class ChatRespondRequest(BaseModel):
    session_id: str
    retrieval_id: str


class FeedbackCandidateRef(BaseModel):
    memory_id: str
    lane: str = ""
    cluster_id: str = ""
    score: float = 0.0
    base_score: float = 0.0
    source: str = ""
    display_text: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    query_text: str
    memory_id: str
    feedback_type: str
    lane: str = ""
    candidate_refs: List[FeedbackCandidateRef] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str = ""
    selected_feedback: Dict[str, Any] = Field(default_factory=dict)
    stored_row: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    retrieval_id: str = ""
    memory_refs: List[MemoryRef] = Field(default_factory=list)
    coarse_memory_refs: List[MemoryRef] = Field(default_factory=list)
    association_memory_refs: List[MemoryRef] = Field(default_factory=list)
    association_tags: List[Dict[str, Any]] = Field(default_factory=list)
    association_trace: Dict[str, Any] = Field(default_factory=dict)
    suppressor_trace: Dict[str, Any] = Field(default_factory=dict)
    history: List[ChatTurn] = Field(default_factory=list)
    retrieval_label: str = ""


class ChatRetrieveResponse(BaseModel):
    session_id: str
    retrieval_id: str
    reply: str = ""
    memory_refs: List[MemoryRef] = Field(default_factory=list)
    coarse_memory_refs: List[MemoryRef] = Field(default_factory=list)
    association_memory_refs: List[MemoryRef] = Field(default_factory=list)
    association_tags: List[Dict[str, Any]] = Field(default_factory=list)
    association_trace: Dict[str, Any] = Field(default_factory=dict)
    suppressor_trace: Dict[str, Any] = Field(default_factory=dict)
    history: List[ChatTurn] = Field(default_factory=list)
    retrieval_label: str = ""


class SessionResponse(BaseModel):
    session_id: str
    history: List[ChatTurn] = Field(default_factory=list)
