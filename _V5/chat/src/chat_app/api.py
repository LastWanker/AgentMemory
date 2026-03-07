from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

V5_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = V5_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agentmemory_v3.association import AssociationGraphView

from .config import load_config
from .models import (
    ChatRequest,
    ChatRespondRequest,
    ChatResponse,
    ChatRetrieveRequest,
    ChatRetrieveResponse,
    FeedbackRequest,
    FeedbackResponse,
    SessionResponse,
)
from .service import ChatService


def create_app() -> FastAPI:
    config = load_config()
    service = ChatService(config)
    graph_view = AssociationGraphView.from_config(config.retrieval_config)
    static_dir = Path(__file__).resolve().parents[2] / "static"

    app = FastAPI(title="AgentMemory V5 Chat")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/graph")
    def graph_page() -> FileResponse:
        return FileResponse(static_dir / "graph.html")

    @app.get("/activation")
    def activation_page() -> FileResponse:
        return FileResponse(static_dir / "activation.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": config.model,
            "retrieval_mode": config.retrieval_mode,
            "retrieval_label": "coarse+association",
            "retrieval_bundle_exists": config.retrieval_bundle.exists(),
            "graph_manifest": graph_view.debug_summary(),
        }

    @app.get("/api/graph-view")
    def graph_lookup(level: str, name: str) -> dict:
        payload = graph_view.lookup(level, name)
        if bool(payload.get("ok")):
            return payload
        detail = payload.get("error")
        status = 404 if detail == "not_found" else 400
        raise HTTPException(status_code=status, detail=payload)

    @app.post("/api/session/new", response_model=SessionResponse)
    def new_session() -> SessionResponse:
        session_id = service.new_session()
        return SessionResponse(session_id=session_id, history=[])

    @app.get("/api/session/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str) -> SessionResponse:
        return SessionResponse(session_id=session_id, history=service.load_session(session_id))

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        try:
            return service.chat(
                request.session_id,
                request.text.strip(),
                request.top_k,
                request.memory_preference_enabled,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat/retrieve", response_model=ChatRetrieveResponse)
    def chat_retrieve(request: ChatRetrieveRequest) -> ChatRetrieveResponse:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        try:
            return service.chat_retrieve(
                request.session_id,
                request.text.strip(),
                request.top_k,
                request.memory_preference_enabled,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat/respond", response_model=ChatResponse)
    def chat_respond(request: ChatRespondRequest) -> ChatResponse:
        if not request.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is empty")
        if not request.retrieval_id.strip():
            raise HTTPException(status_code=400, detail="retrieval_id is empty")
        try:
            return service.chat_respond(request.session_id.strip(), request.retrieval_id.strip())
        except KeyError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        if not request.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is empty")
        if not request.query_text.strip():
            raise HTTPException(status_code=400, detail="query_text is empty")
        if not request.memory_id.strip():
            raise HTTPException(status_code=400, detail="memory_id is empty")
        feedback_type = request.feedback_type.strip().lower()
        if feedback_type not in {"unrelated", "toforget"}:
            raise HTTPException(status_code=400, detail="feedback_type must be unrelated or toforget")
        payload = service.record_feedback(
            session_id=request.session_id.strip(),
            query_text=request.query_text.strip(),
            memory_id=request.memory_id.strip(),
            feedback_type=feedback_type,
            lane=request.lane.strip(),
            candidate_refs=[item.model_dump() for item in request.candidate_refs],
        )
        return FeedbackResponse.model_validate(payload)

    return app
