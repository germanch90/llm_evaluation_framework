"""FastAPI entrypoint for the RAG service."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.rag_pipeline import RAGPipeline, build_pipeline

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(..., description="Natural language question from the user")
    top_k: int = Field(
        5, ge=1, le=20, description="Maximum number of chunks to retrieve"
    )


class Chunk(BaseModel):
    """Shape of retrieved context chunks."""

    id: str
    score: float
    content: str
    metadata: Dict[str, Any] | None = None


class QueryResponse(BaseModel):
    """Outgoing response payload."""

    answer: str
    chunks: List[Chunk]
    metrics_ms: Dict[str, float] = Field(
        default_factory=dict, description="Latency metrics per pipeline stage"
    )
    generated_at: datetime


class HealthResponse(BaseModel):
    """Health-check payload for observability tooling."""

    status: str
    message: str


app = FastAPI(
    title="RAG Portfolio API",
    version="0.1.0",
    description="Minimal FastAPI surface for the RAG reference implementation.",
)

try:
    pipeline = build_pipeline()
except Exception as exc:  # pragma: no cover - defensive logging
    logger.warning("Unable to initialize RAG pipeline: %s", exc)
    pipeline = None


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Lightweight readiness probe used by Docker healthcheck."""
    return HealthResponse(status="ok", message="Service healthy")


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Handle a RAG query request."""

    cleaned = request.query.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Query must not be empty")

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not available. Check server logs for details.",
        )

    try:
        result = await pipeline.query_async(cleaned, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Pipeline query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Pipeline query failed") from exc

    chunks = [
        Chunk(
            id=ctx.id,
            score=ctx.score,
            content=ctx.content,
            metadata=ctx.metadata or {},
        )
        for ctx in result.contexts
    ]

    return QueryResponse(
        answer=result.answer,
        chunks=chunks,
        metrics_ms=result.metrics_ms,
        generated_at=datetime.utcnow(),
    )
