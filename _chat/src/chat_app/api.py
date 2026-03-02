from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .models import ChatRequest, ChatResponse, SessionResponse
from .service import ChatService


def create_app() -> FastAPI:
    config = load_config()
    service = ChatService(config)
    static_dir = Path(__file__).resolve().parents[2] / "static"

    app = FastAPI(title="AgentMemory Local Chat")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": config.model,
            "retrieval_mode": config.retrieval_mode,
            "retrieval_bundle_exists": config.retrieval_bundle.exists(),
        }

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
                coarse_only=request.coarse_only,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app
