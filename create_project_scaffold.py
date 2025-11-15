#!/usr/bin/env python3
"""
RAG Portfolio Project Scaffold Generator
Creates complete directory structure and starter files for RAG implementation
"""

import os
from pathlib import Path
from typing import Dict

def create_directory_structure(base_path: Path) -> None:
    """Create all project directories."""
    directories = [
        "src/ingestion",
        "src/processing",
        "src/retrieval",
        "src/generation",
        "src/pipeline",
        "src/api",
        "tests/unit",
        "tests/integration",
        "tests/evaluation",
        "data/documents",
        "data/vector_db",
        "data/kg-rag-dataset",
        "logs",
        "docs",
        "notebooks",
        "config",
        "scripts",  # Added scripts directory
    ]
    
    for directory in directories:
        path = base_path / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
        
        # Create __init__.py for Python packages
        if directory.startswith("src/") or directory.startswith("tests/"):
            init_file = path / "__init__.py"
            init_file.touch()

def create_file(path: Path, content: str, base_path: Path) -> None:
    """Create a file with given content."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n")
    try:
        rel_path = path.relative_to(base_path)
    except ValueError:
        rel_path = path.name
    print(f"✓ Created file: {rel_path}")

def generate_files(base_path: Path) -> None:
    """Generate all starter files."""
    
    files_content: Dict[str, str] = {
        # Root files
        "README.md": """# RAG Portfolio Project

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
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
""",
        
        ".env.example": """# LLM Configuration
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512

# Embedding Configuration
EMBEDDING_MODEL=nomic-embed-text

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/vector_db
CHROMA_COLLECTION_NAME=rag_documents

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=51

# Retrieval Configuration
TOP_K=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag.log
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Project specific
data/documents/*.pdf
data/vector_db/
logs/*.log
.env

# Testing
.pytest_cache/
.coverage
htmlcov/
reports/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
*.env.local
""",
        
        "requirements.txt": """# Core dependencies
langchain==0.3.7
langchain-community==0.3.5
chromadb==0.5.20
pypdf==5.1.0
python-dotenv==1.0.1
pydantic==2.10.3
pydantic-settings==2.6.1

# API
fastapi==0.115.6
uvicorn[standard]==0.32.1

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
deepeval==1.4.7

# Utilities
python-multipart==0.0.20
pyyaml==6.0.2
requests==2.32.3
tqdm==4.67.1

# Optional: Streamlit UI
# streamlit==1.40.2
""",
        
        "requirements-dev.txt": """# Development dependencies
-r requirements.txt

# Code quality
black==24.10.0
isort==5.13.2
flake8==7.1.1
mypy==1.13.0
pylint==3.3.2

# Testing
pytest-cov==6.0.0
pytest-html==4.1.1

# Documentation
mkdocs==1.6.1
mkdocs-material==9.5.47
""",
        
        "docker-compose.yml": """version: '3.8'

services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - CHROMA_PERSIST_DIRECTORY=/app/data/vector_db
    env_file:
      - .env
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    depends_on:
      - chromadb
    networks:
      - rag-network

  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    ports:
      - "8001:8000"
    volumes:
      - ./data/vector_db:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
""",
        
        "Dockerfile": """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p data/documents data/vector_db logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
        
        "config/config.yaml": """llm:
  model: "llama3.1:8b"
  temperature: 0.1
  max_tokens: 512
  timeout: 30

embeddings:
  model: "nomic-embed-text"
  batch_size: 100

chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 51
  separators:
    - "\\n\\n"
    - "\\n"
    - ". "
    - " "
    - ""

retrieval:
  top_k: 5
  score_threshold: 0.5

vector_db:
  type: "chromadb"
  collection_name: "rag_documents"
  distance_metric: "cosine"

evaluation:
  metrics:
    answer_relevancy:
      threshold: 0.7
    faithfulness:
      threshold: 0.8
    contextual_relevancy:
      threshold: 0.6
  
  performance:
    max_latency_seconds: 5.0

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rag.log"
""",
        
        "config/prompts.yaml": """system_prompt: |
  You are a helpful assistant that answers questions based on provided context.
  Always ground your answers in the given context and cite sources when possible.

rag_prompt_template: |
  Context information is below:
  ---------------------
  {context}
  ---------------------
  
  Given the context information above, please answer the following question.
  If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided context."
  
  Question: {query}
  
  Answer:

no_context_prompt: |
  I apologize, but I couldn't find relevant information in the document corpus to answer your question.
  Please try rephrasing your question or ask about a different topic.
""",
        
        "pytest.ini": """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing

markers =
    unit: Unit tests
    integration: Integration tests
    evaluation: DeepEval evaluation tests
    slow: Slow running tests
""",
        
        ".dockerignore": """**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
.Python
venv/
env/
.venv
.git
.gitignore
.pytest_cache
.coverage
htmlcov/
*.log
.env
.env.local
README.md
docs/
notebooks/
tests/
data/documents/*.pdf
data/vector_db/
""",
        
        "Makefile": """.PHONY: help setup test clean docker-build docker-up docker-down lint format

help:
\t@echo "Available commands:"
\t@echo "  setup          - Set up development environment"
\t@echo "  test           - Run all tests"
\t@echo "  test-eval      - Run evaluation tests only"
\t@echo "  clean          - Clean up generated files"
\t@echo "  docker-build   - Build Docker images"
\t@echo "  docker-up      - Start Docker containers"
\t@echo "  docker-down    - Stop Docker containers"
\t@echo "  lint           - Run code linters"
\t@echo "  format         - Format code"
\t@echo "  download-data  - Download KG-RAG dataset"

setup:
\tpython -m venv venv
\t. venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

test:
\tpytest

test-eval:
\tpytest tests/evaluation/ -v

clean:
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name "*.pyc" -delete
\trm -rf .pytest_cache htmlcov/ .coverage
\trm -rf logs/*.log

docker-build:
\tdocker-compose build

docker-up:
\tdocker-compose up -d

docker-down:
\tdocker-compose down

lint:
\tflake8 src/ tests/
\tmypy src/
\tpylint src/

format:
\tblack src/ tests/
\tisort src/ tests/

download-data:
\tpython scripts/download_dataset.py
""",
        
        "scripts/download_dataset.py": """#!/usr/bin/env python3
\"\"\"
Download Docugami KG-RAG dataset
\"\"\"
import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url: str, destination: Path) -> None:
    \"\"\"Download file with progress bar.\"\"\"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def main():
    \"\"\"Download and extract KG-RAG dataset.\"\"\"
    # GitHub repository info
    repo_url = "https://github.com/docugami/KG-RAG-datasets"
    
    print("=" * 60)
    print("Docugami KG-RAG Dataset Downloader")
    print("=" * 60)
    print(f"\\nDataset repository: {repo_url}")
    print("\\nPlease follow these steps:")
    print("1. Visit the repository URL above")
    print("2. Clone or download the datasets you need")
    print("3. Place them in: data/kg-rag-dataset/")
    print("\\nRecommended datasets:")
    print("  - Agriculture")
    print("  - Automotive") 
    print("  - Biomedical")
    print("  - Finance")
    print("  - Legal")
    print("\\nEach dataset contains:")
    print("  - Documents (PDFs)")
    print("  - Questions with ground truth answers")
    print("  - Difficulty levels (single-chunk to multi-doc)")
    print("=" * 60)
    
    # Create directory
    dataset_dir = Path("data/kg-rag-dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\\n✓ Created directory: {dataset_dir}")
    print("\\nManual setup required - see instructions above.")

if __name__ == "__main__":
    main()
""",
        
        # Source files start here
        "src/ingestion/document_loader.py": """\"\"\"
Document loading and preprocessing.
\"\"\"
import logging
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class Document:
    \"\"\"Represents a document with content and metadata.\"\"\"
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    \"\"\"Load documents from various sources.\"\"\"
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        \"\"\"
        Load a PDF file and extract text with page metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects, one per page
        \"\"\"
        documents = []
        
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_num,
                            "total_pages": len(reader.pages)
                        }
                    )
                    documents.append(doc)
                    
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: Path) -> List[Document]:
        \"\"\"
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all documents from all PDFs
        \"\"\"
        all_documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            documents = self.load_pdf(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Loaded total of {len(all_documents)} documents")
        return all_documents
""",
        
        "src/processing/chunking.py": """\"\"\"
Text chunking strategies.
\"\"\"
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunk:
    \"\"\"Represents a text chunk with metadata.\"\"\"
    
    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: str):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
    
    def __repr__(self) -> str:
        return f"TextChunk(id={self.chunk_id}, length={len(self.text)})"


class DocumentChunker:
    \"\"\"Chunk documents into smaller pieces for retrieval.\"\"\"
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51):
        \"\"\"
        Initialize chunker with recursive character text splitting.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
        \"\"\"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
        
        logger.info(
            f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_documents(self, documents: List[Any]) -> List[TextChunk]:
        \"\"\"
        Chunk a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of TextChunk objects
        \"\"\"
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            texts = self.splitter.split_text(document.content)
            
            for chunk_idx, text in enumerate(texts):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                
                # Preserve original metadata and add chunk info
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(texts)
                })
                
                chunk = TextChunk(
                    text=text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                )
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        self._log_chunk_statistics(all_chunks)
        
        return all_chunks
    
    def _log_chunk_statistics(self, chunks: List[TextChunk]) -> None:
        \"\"\"Log statistics about chunk sizes.\"\"\"
        if not chunks:
            return
        
        sizes = [len(chunk.text) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        logger.info(
            f"Chunk statistics - Avg: {avg_size:.0f}, "
            f"Min: {min_size}, Max: {max_size}"
        )
""",
        
        "src/retrieval/embeddings.py": """\"\"\"
Embedding generation using Ollama.
\"\"\"
import logging
from typing import List
import requests

logger = logging.getLogger(__name__)


class EmbeddingModel:
    \"\"\"Generate embeddings using Ollama.\"\"\"
    
    def __init__(self, model_name: str = "nomic-embed-text", ollama_host: str = "http://localhost:11434"):
        \"\"\"
        Initialize embedding model.
        
        Args:
            model_name: Name of the Ollama embedding model
            ollama_host: URL of Ollama server
        \"\"\"
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/embeddings"
        
        logger.info(f"Initialized embedding model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        \"\"\"
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        \"\"\"
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["embedding"]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        \"\"\"
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        \"\"\"
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return embeddings
""",
    }
    
    # Create all files
    for file_path, content in files_content.items():
        create_file(base_path / file_path, content, base_path)

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("RAG Portfolio Project Scaffold Generator")
    print("="*60 + "\n")
    
    # Get project name
    project_name = input("Enter project name (default: rag-portfolio): ").strip()
    if not project_name:
        project_name = "rag-portfolio"
    
    base_path = Path.cwd() / project_name
    
    if base_path.exists():
        response = input(f"\n⚠️  Directory '{project_name}' already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print(f"\nCreating project in: {base_path}\n")
    
    # Create structure
    create_directory_structure(base_path)
    print()
    generate_files(base_path)
    
    print("\n" + "="*60)
    print("✅ Project scaffold created successfully!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. cd {project_name}")
    print(f"2. cp .env.example .env")
    print(f"3. Install Ollama and pull models:")
    print(f"   - ollama pull llama3.1:8b")
    print(f"   - ollama pull nomic-embed-text")
    print(f"4. Create virtual environment:")
    print(f"   - python -m venv venv")
    print(f"   - source venv/bin/activate")
    print(f"5. Install dependencies:")
    print(f"   - pip install -r requirements.txt")
    print(f"6. Download KG-RAG dataset:")
    print(f"   - python scripts/download_dataset.py")
    print(f"7. Start building! 🚀")
    print("\n")

if __name__ == "__main__":
    main()