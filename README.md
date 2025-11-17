# RAG Portfolio Project

Production-quality Retrieval-Augmented Generation system with comprehensive testing.

## Architecture

```
Document (PDF) → Chunking → Embedding → Vector Store (ChromaDB)
                                              ↓
User Query → Query Embedding → Similarity Search → Top-K Chunks
                                                        ↓
                                            LLM (Llama 3.1 8B) → Answer
```

## Tech Stack

- **LLM**: Llama 3.1 8B (via Ollama)
- **Embeddings**: nomic-embed-text (via Ollama)
- **Vector DB**: ChromaDB
- **Chunking**: RecursiveCharacterTextSplitter (512 tokens, 10% overlap)
- **Testing**: deepeval + pytest
- **API**: FastAPI
- **Dataset**: Docugami KG-RAG
- **Deployment**: Docker Compose

## Setup

### Prerequisites

1. Install Docker and Docker Compose
2. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
3. Pull required models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

### Installation

1. Clone the repository.
2. Copy the environment template: `cp .env.example .env`
3. Build and launch the stack:
   ```bash
   docker compose up -d --build
   ```

### Populate the Dataset

Download the Docugami KG-RAG datasets into `data/kg-rag-dataset/`:

```bash
docker compose exec rag-api python scripts/download_dataset.py --overwrite
```

This script downloads the upstream GitHub archive and extracts all datasets by default. Use `--datasets` to limit the extraction (e.g., `--datasets us-fed-agency-reports`).

### Local Development (optional)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

### Ingestion

1. Prepare a directory of PDFs (e.g., copy a subset into `data/demo-docs/`).
2. Run the ingestion CLI inside the container:
   ```bash
   docker compose exec rag-api python -m src.pipeline.cli ingest --input data/demo-docs
   ```
   This loads documents, chunks them, generates embeddings via Ollama, and stores them in Chroma (`./data/vector_db`).

### Query the Pipeline

#### CLI

```bash
docker compose exec rag-api python -m src.pipeline.cli query \
  --question "What priorities are highlighted in the Department of Labor report?" \
  --top-k 3
```

#### FastAPI

- Health check: `docker compose exec rag-api curl -s http://localhost:8000/health`
- Query endpoint:
  ```bash
  docker compose exec rag-api curl -s http://localhost:8000/query \
    -H 'Content-Type: application/json' \
    -d '{"query": "What are the DOL goals?", "top_k": 2}' | jq
  ```

### Open WebUI (Optional)

1. Ensure `ollama` and `ollama-webui` services are running (`docker compose up -d`).
2. Visit `http://localhost:8081`, log in with the credentials defined in `.env` (`DEFAULT_USER`, `DEFAULT_PASSWORD`).
3. The WebUI is pointed at the internal Ollama host (`OLLAMA_BASE_URLS=http://ollama:11434`). Use it to issue ad hoc queries; for full RAG context, call the FastAPI `/query` endpoint via custom connectors or the CLI.

### Prepare Evaluation Test Data

Generate balanced DeepEval test cases from the datasets:

```bash
docker compose exec rag-api env PYTHONPATH=/app python scripts/prepare_test_data.py \
  --dataset-root data/kg-rag-dataset \
  --output data/test_cases.json \
  --per-difficulty 8 --max-total 24
```

### Run Tests

```bash
# Unit + integration suite
docker compose run --rm test

# Specific test module
docker compose run --rm test pytest tests/unit/test_llm_client.py
```

## Project Structure

```
├── config/                  # YAML configs and prompts
├── data/                    # Datasets, vector DB, test_cases.json (ignored from git)
├── docs/                    # Architecture and documentation
├── scripts/                 # Dataset download & test data preparation
├── src/
│   ├── api/                 # FastAPI app
│   ├── evaluation/          # Test data builder utilities
│   ├── generation/          # LLM client & prompts
│   ├── ingestion/           # PDF loader
│   ├── pipeline/            # RAG pipeline + CLI
│   ├── processing/          # Chunking
│   └── retrieval/           # Embeddings + vector store
└── tests/                   # Pytest suites
```

## Testing Methodology

The project leverages `pytest` and `deepeval` for RAG-specific metrics:

- **Answer Relevancy** ≥ 0.7
- **Faithfulness** ≥ 0.8
- **Contextual Relevancy** ≥ 0.6
- **Custom metrics**: Latency, citation accuracy, no-answer detection

## Future Enhancements

- Semantic/hybrid search and reranking
- Streamlit or richer UI
- `/ingest` API endpoint + upload workflow
- Automated DeepEval suite and CI integration

## License

MIT License
