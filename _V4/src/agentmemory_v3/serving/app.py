from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever


class RetrieveRequestModel(BaseModel):
    query: str
    top_k: int = 5
    coarse_only: bool = False


def create_app(config_path: str) -> FastAPI:
    retriever = HybridRetriever.from_config(config_path)
    app = FastAPI(title="AgentMemory V4 Retriever")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "memory_count": len(retriever.artifacts.memory_rows),
            "cluster_count": len(retriever.artifacts.clusters),
        }

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequestModel) -> dict:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query is empty")
        hits, trace = retriever.retrieve(
            request.query.strip(),
            request.top_k,
            coarse_only=request.coarse_only,
        )
        return {"hits": [hit.__dict__ for hit in hits], "trace": trace}

    return app
