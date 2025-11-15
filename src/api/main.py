"""FastAPI entrypoint for the RAG service."""

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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


class QueryResponse(BaseModel):
    """Outgoing response payload."""

    answer: str
    chunks: List[Chunk]
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


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Lightweight readiness probe used by Docker healthcheck."""
    return HealthResponse(status="ok", message="Service healthy")


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Handle a RAG query request.

    The actual retrieval / generation logic has not been wired up yet, so we
    return a deterministic placeholder response. This keeps the container
    bootable until the rest of the pipeline is implemented.
    """

    cleaned = request.query.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Query must not be empty")

    placeholder_chunk = Chunk(
        id="placeholder-0",
        score=1.0,
        content="No pipeline implemented yet. Replace with retrieved chunk.",
    )

    return QueryResponse(
        answer=(
            "RAG pipeline not implemented yet. "
            "Wire up src.pipeline components to generate real answers."
        ),
        chunks=[placeholder_chunk] * min(request.top_k, 1),
        generated_at=datetime.utcnow(),
    )
