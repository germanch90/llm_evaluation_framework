"""
Core RAG pipeline implementation.

Provides ingestion and query flows that connect document loaders, chunkers,
embeddings, vector store retrieval, and the LLM generation client. Includes
stage-level logging, error handling, and latency metrics that can be reused by
CLI tooling or the FastAPI surface.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.llm_client import AsyncOllamaClient, LLMResponse
from src.generation.prompts import PromptManager
from src.ingestion.document_loader import DocumentLoader
from src.processing.chunking import DocumentChunker, TextChunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaDBVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Chunk returned from vector retrieval."""

    id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Normalized query result returned by the pipeline."""

    question: str
    answer: str
    contexts: List[RetrievedChunk]
    llm_response: Optional[LLMResponse]
    metrics_ms: Dict[str, float]


class RAGPipeline:
    """End-to-end RAG pipeline with ingestion and query utilities."""

    def __init__(
        self,
        *,
        document_loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[ChromaDBVectorStore] = None,
        llm_client: Optional[AsyncOllamaClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        top_k: int = 5,
    ) -> None:
        self.document_loader = document_loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or ChromaDBVectorStore()
        self.llm_client = llm_client or AsyncOllamaClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self.system_prompt = self.prompt_manager.format_system_prompt()
        self.top_k = top_k

        logger.info(
            "RAGPipeline initialized: top_k=%s, model=%s, vector_store=%s",
            self.top_k,
            self.llm_client.model_name,
            self.vector_store.collection_name,
        )

    def ingest_directory(self, directory: Path) -> Dict[str, Any]:
        """
        Ingest all PDF documents in a directory.

        Returns statistics about the ingestion job.
        """
        metrics: Dict[str, float] = {}
        overall_start = time.perf_counter()
        directory = Path(directory)

        try:
            load_start = time.perf_counter()
            documents = self.document_loader.load_directory(directory)
            metrics["load_ms"] = (time.perf_counter() - load_start) * 1000
            if not documents:
                logger.warning("No documents found in directory %s", directory)
                return {
                    "documents": 0,
                    "chunks": 0,
                    "embeddings": 0,
                    "metrics_ms": metrics,
                }

            chunk_start = time.perf_counter()
            chunks = self.chunker.chunk_documents(documents)
            metrics["chunk_ms"] = (time.perf_counter() - chunk_start) * 1000

            embed_start = time.perf_counter()
            embeddings = self.embedding_model.embed_batch(
                [chunk.text for chunk in chunks]
            )
            metrics["embed_ms"] = (time.perf_counter() - embed_start) * 1000

            store_start = time.perf_counter()
            self.vector_store.add(
                embeddings=embeddings,
                metadatas=[chunk.metadata for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
            )
            metrics["store_ms"] = (time.perf_counter() - store_start) * 1000

            metrics["total_ms"] = (time.perf_counter() - overall_start) * 1000

            stats = {
                "documents": len(documents),
                "chunks": len(chunks),
                "embeddings": len(embeddings),
                "metrics_ms": metrics,
            }
            logger.info(
                "Ingestion complete: docs=%s chunks=%s embeddings=%s total_ms=%.1f",
                stats["documents"],
                stats["chunks"],
                stats["embeddings"],
                metrics.get("total_ms", 0.0),
            )
            return stats
        except Exception as exc:
            logger.exception("Ingestion failed for %s: %s", directory, exc)
            raise

    async def query_async(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Run the full RAG flow asynchronously for a given question.
        """
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        metrics: Dict[str, float] = {}
        overall_start = time.perf_counter()

        # Embed the query
        embed_start = time.perf_counter()
        query_embedding = self.embedding_model.embed_text(question)
        metrics["query_embed_ms"] = (time.perf_counter() - embed_start) * 1000

        # Retrieve relevant chunks
        retrieve_start = time.perf_counter()
        results = self.vector_store.query(
            query_embeddings=[query_embedding], top_k=top_k or self.top_k
        )
        metrics["retrieve_ms"] = (time.perf_counter() - retrieve_start) * 1000

        contexts = self._build_contexts(results)
        context_text = "\n\n".join(
            context.content for context in contexts
        ).strip()

        if not context_text:
            logger.warning("No relevant context retrieved for question: %s", question)
            answer = self.prompt_manager.format_no_context_prompt(question)
            return RAGResult(
                question=question,
                answer=answer,
                contexts=[],
                llm_response=None,
                metrics_ms={**metrics, "total_ms": (time.perf_counter() - overall_start) * 1000},
            )

        # Generate answer
        generate_start = time.perf_counter()
        llm_response = await self.llm_client.generate_with_context(
            query=question,
            context=context_text,
            system_prompt=self.system_prompt,
        )
        metrics["generate_ms"] = (time.perf_counter() - generate_start) * 1000
        metrics["total_ms"] = (time.perf_counter() - overall_start) * 1000

        return RAGResult(
            question=question,
            answer=llm_response.answer,
            contexts=contexts,
            llm_response=llm_response,
            metrics_ms=metrics,
        )

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Convenience wrapper for synchronous code paths (e.g., CLI).
        """
        return asyncio.run(self.query_async(question=question, top_k=top_k))

    def _build_contexts(self, results: Dict[str, Any]) -> List[RetrievedChunk]:
        """Normalize raw Chroma results into RetrievedChunk objects."""
        if not results or not results.get("documents"):
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: List[RetrievedChunk] = []
        for doc, meta, chunk_id, distance in zip(
            documents, metadatas, ids, distances
        ):
            if doc is None:
                continue
            # Convert distance to similarity score; guard against None
            score = 1.0 - float(distance or 0.0)
            chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    score=score,
                    content=doc,
                    metadata=meta or {},
                )
            )
        return chunks


def build_pipeline() -> RAGPipeline:
    """Factory helper used by API or CLI entrypoints."""
    return RAGPipeline()
