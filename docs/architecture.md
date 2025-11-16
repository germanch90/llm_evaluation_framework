# RAG Portfolio Architecture (Phase 1)

This document summarizes how the current system is wired together after completing Phase 1.3 (RAG pipeline integration).

## High-Level Flow

```
        ┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐
PDFs → │ Ingestion   │  →   │ Chunker    │ → Emb │ Embeddings │ →   │ ChromaDB   │
        │ (Loader)   │       │ (LangChain)│       │ (Ollama)   │       │ Vector DB  │
        └────────────┘       └────────────┘       └────────────┘       └────┬──────┘
                                                                            │
                                                                            ▼
                                                            ┌────────────────────────┐
Query ─────────────→ Embed (Ollama) ──→ Retrieve top-k chunks ──→ LLM generation (AsyncOllamaClient)
                                                            └────────────────────────┘
                                                                             │
                                                                             ▼
                                             Answer + sources returned via CLI or FastAPI `/query`
```

## Container Topology

- **ollama**: GPU-enabled Llama 3.1 and nomic-embed-text server (port 7869). Persistent model cache mounted from `${OLLAMA_MODELS_DIR}`. No external traffic except within the Docker network.
- **chromadb**: Persistent Chroma vector store (port 8001). Volume `./data/vector_db` keeps embeddings.
- **rag-api**:
  - Runs FastAPI (`src/api/main.py`) + CLI utilities.
  - Mounts host `./data`, `./config`, `./logs`.
  - Uses `OLLAMA_HOST` to talk to the Ollama service and `CHROMA_PERSIST_DIRECTORY=/app/data/vector_db`.
  - Contains the RAG pipeline, ingestion CLI, download script, and serves the `/query` and `/health` endpoints.
- **ollama-webui**: Optional Open WebUI surface on port 8081, pointing to `http://ollama:11434`, protected with credentials from `.env`.
- **test**: Ephemeral container used for running `pytest` with the same code/volumes to keep dev and CI in sync.

All services share the `rag-network` bridge and mount the host data/log directories so artifacts persist.

## Core Components

- **Document Loader (`src/ingestion/document_loader.py`)**: Extracts text per page with metadata (source path, page number).
- **Chunker (`src/processing/chunking.py`)**: Uses LangChain’s `RecursiveCharacterTextSplitter` (512 chars with 51 overlap) and logs stats.
- **Embeddings (`src/retrieval/embeddings.py`)**: Calls Ollama embeddings API, batching through `embed_batch` and honoring `OLLAMA_HOST`.
- **Vector Store (`src/retrieval/vector_store.py`)**: Wrapper around Chroma add/query/upsert/delete operations with logging and metadata filtering.
- **LLM Client (`src/generation/llm_client.py`)**: Async interface to Ollama text generation with retries, token accounting, prompt templating.
- **Pipeline (`src/pipeline/rag_pipeline.py`)**:
  - `ingest_directory`: load → chunk → embed → store with per-stage latency metrics.
  - `query_async`: embed question, retrieve top-k chunks, format prompts via `PromptManager`, call LLM, and return structured results.
  - CLI wrapper (`src/pipeline/cli.py`) exposes `ingest` and `query` commands for demos.
- **API (`src/api/main.py`)**: `/query` calls the shared pipeline (same code path as CLI) and returns answers + chunk metadata + timing.

## Data Pipeline

1. **Download**: `scripts/download_dataset.py` pulls the Docugami KG-RAG repo zip and extracts datasets under `data/kg-rag-dataset/`. Runs equally on host or inside `rag-api`.
2. **Ingest**:
   - `docker compose exec rag-api python -m src.pipeline.cli ingest --input data/demo-docs`
   - Or point to any `data/kg-rag-dataset/.../docs` folder.
   - Embeddings persisted in `./data/vector_db`.
3. **Query**:
   - CLI: `docker compose exec rag-api python -m src.pipeline.cli query --question "..." --top-k 3`
   - API: `curl` or Open WebUI hitting FastAPI `/query`.

## Observability / Metrics

- Stage-level latency (`metrics_ms`) recorded for ingestion and queries (embed/retrieve/generate/total) and surfaced via CLI/API responses.
- Structured logging at INFO/WARNING/ERROR with component names (ingestion, chunking, embeddings, vector store, pipeline, API).
- `/health` endpoint used by Docker health checks.

## Security Considerations

- Ollama and Chroma exposed only within Docker network by default; host ports opened only if needed (Ollama 7869, API 8000, WebUI 8081).
- WebUI requires credentials defined in `.env`. FastAPI validates non-empty queries and wraps pipeline errors with HTTP status codes.
- Download script writes under `data/kg-rag-dataset/`, which is `.gitignore`’d to prevent accidental check-ins of PDFs.

