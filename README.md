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

1. Clone the repository
2. Copy environment file: `cp .env.example .env`
3. Build and run with Docker:
   ```bash
   docker-compose up --build
   ```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download KG-RAG dataset
python scripts/download_dataset.py
```

## Usage

### CLI Interface

```bash
# Ingest documents
python -m src.pipeline.ingest --input data/documents/

# Query the system
python -m src.pipeline.query "What is the main topic?"
```

### API Interface

```bash
# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Query via curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is X?", "top_k": 5}'
```

### Run Tests

```bash
# Run all tests
pytest

# Run evaluation tests only
pytest tests/evaluation/

# Generate test report
pytest tests/evaluation/ --html=reports/test_report.html
```

## Project Structure

```
rag-portfolio/
├── src/                    # Source code
│   ├── ingestion/         # Document loading
│   ├── processing/        # Chunking strategies
│   ├── retrieval/         # Embeddings and vector store
│   ├── generation/        # LLM integration
│   ├── pipeline/          # Main RAG pipeline
│   └── api/               # FastAPI application
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── evaluation/       # DeepEval tests
├── data/                  # Data directory
├── docs/                  # Documentation
└── config/               # Configuration files
```

## Testing Methodology

This project uses deepeval for RAG-specific evaluation:

- **Answer Relevancy**: Does the answer address the query?
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Contextual Relevancy**: Are retrieved chunks relevant?
- **Custom Metrics**: Latency, citation accuracy, no-answer detection

Target thresholds:
- Answer Relevancy: ≥ 0.7
- Faithfulness: ≥ 0.8
- Contextual Relevancy: ≥ 0.6
- Latency: < 5 seconds

## Performance Baseline

See [docs/baseline_results.md](docs/baseline_results.md) for detailed metrics.

## Future Enhancements

- [ ] Semantic chunking
- [ ] Hybrid search (vector + keyword)
- [ ] Re-ranking layer
- [ ] Query rewriting
- [ ] Streaming responses
- [ ] Multi-document support
- [ ] Chat history management

## License

MIT License
