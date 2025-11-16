# RAG Portfolio Project - Requirements Plan

## Project Goal
Build a production-quality RAG system with comprehensive testing using deepeval framework, demonstrating accuracy-focused implementation with Docugami KG-RAG datasets.

---

## Tech Stack (Final)

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| LLM | Llama 3.1 8B (Ollama) | Local, cost-free, runs on RTX 3090 Ti |
| Embeddings | nomic-embed-text (Ollama) | High-quality, free, local |
| Vector DB | ChromaDB | Embedded mode, simple setup |
| Chunking | RecursiveCharacterTextSplitter | Reliable baseline (512 tokens, 10% overlap) |
| Testing | deepeval + pytest | RAG-specific metrics |
| API | FastAPI | Modern, async, auto-docs |
| Dataset | Docugami KG-RAG | Realistic multi-doc scenarios |
| Deployment | Docker Compose | Reproducible environment |

---

## Component: Core RAG Pipeline

### 1.1 Vector Store Implementation
- [x] Implement ChromaDB wrapper with CRUD operations
- [x] Configure collection with cosine similarity
- [x] Add batch insertion for efficiency
- [x] Implement persistence to disk
- [x] Add metadata filtering capability

### 1.2 LLM Client
- [x] Create Ollama client for llama3.1:8b
- [x] Implement prompt template system
- [x] Add temperature and token limit controls
- [x] Handle timeouts and retries
- [x] Track token usage

### 1.3 RAG Pipeline Integration
- [x] Build end-to-end pipeline class
- [x] Chain: query → embed → retrieve → generate
- [x] Add logging at each stage
- [x] Implement error handling
- [x] Track latency metrics
- [x] Create CLI interface for testing

### 1.4 Document Ingestion Flow
- [x] Create ingestion script to process KG-RAG PDFs
- [x] Load → Chunk → Embed → Store pipeline
- [x] Log statistics (docs, chunks, embeddings)
- [x] Handle batch processing

**Deliverable**: Working RAG system that answers questions from ingested documents

---

## Component: DeepEval Test Suite

### 2.1 Test Data Preparation
- [ ] Parse KG-RAG dataset Q&A files
- [ ] Convert to test case format (JSON)
- [ ] Select 20-30 questions across difficulty levels:
  - Single-doc, single-chunk (easy)
  - Single-doc, multi-chunk (medium)
  - Multi-doc (hard)
- [ ] Include ground truth answers
- [ ] Add edge cases (unanswerable questions)

### 2.2 Core Metrics Implementation
- [ ] **Answer Relevancy** (threshold: 0.7)
  - Does answer address the query?
- [ ] **Faithfulness** (threshold: 0.8)
  - Is answer grounded in context?
- [ ] **Contextual Relevancy** (threshold: 0.6)
  - Are retrieved chunks relevant?

### 2.3 Custom Metrics
- [ ] **Latency**: End-to-end query time (< 5s target)
- [ ] **Citation Accuracy**: Verify sources exist
- [ ] **No-Answer Detection**: Correctly identifies unanswerable

### 2.4 Test Automation
- [ ] Create pytest test suite
- [ ] Integrate deepeval metrics
- [ ] Generate HTML test reports
- [ ] Document baseline results
- [ ] Set up CI/CD (GitHub Actions)

**Deliverable**: Automated test suite with pass/fail criteria and baseline metrics

---

## Component: API & Interface

### 3.1 FastAPI Implementation
- [x] Implement `/query` endpoint
  - Input: query string, optional top_k
  - Output: answer, sources, metadata, latency
- [x] Add `/health` endpoint
- [ ] Add `/ingest` endpoint (upload PDFs)
- [x] Implement request validation (Pydantic)
- [ ] Add OpenAPI documentation
- [ ] Configure CORS

### 3.2 Optional UI
- [ ] Create Streamlit interface
  - Query input
  - Answer display with sources
  - Document upload
  - Performance metrics

### 3.3 Monitoring & Logging
- [ ] Structured logging (INFO/WARNING/ERROR)
- [ ] Log query/response pairs
- [ ] Track performance metrics
- [ ] Error tracking and reporting

**Deliverable**: Production-ready API with documentation

---

## Component: Docker Deployment

### 4.1 Container Setup
- [ ] Test Docker Compose configuration
- [ ] Verify ChromaDB service connectivity
- [ ] Configure Ollama host access from container
- [ ] Volume mounts for data persistence
- [ ] Health checks for all services

### 4.2 Documentation
- [ ] Update README with Docker instructions
- [ ] Document architecture with diagram
- [ ] Create setup guide
- [ ] Document test results and metrics
- [ ] Add troubleshooting section

**Deliverable**: Fully containerized, reproducible system

---

## Success Criteria

### Functional Requirements
✅ RAG pipeline answers ≥80% of test queries correctly  
✅ DeepEval metrics meet thresholds:
   - Answer Relevancy ≥ 0.7
   - Faithfulness ≥ 0.8
   - Contextual Relevancy ≥ 0.6  
✅ API response time < 5 seconds  
✅ Handles error cases gracefully  

### Code Quality
✅ Type hints on all functions  
✅ Docstrings on public methods  
✅ Separation of concerns (clean architecture)  
✅ Configuration-driven (no hardcoded values)  
✅ Test coverage ≥ 70%  

### Documentation
✅ Setup works on fresh machine  
✅ Architecture clearly explained  
✅ Test results documented with analysis  
✅ Known limitations acknowledged  

### Portfolio Value
✅ Demonstrates RAG fundamentals  
✅ Shows testing methodology  
✅ Production considerations (Docker, API, monitoring)  
✅ Accuracy-focused implementation  
✅ Clean, maintainable code  

---

## Timeline Estimate

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Core Pipeline | 6-8 hours |
| Phase 2: Testing Suite | 4-6 hours |
| Phase 3: API & Interface | 3-4 hours |
| Phase 4: Docker & Docs | 2-3 hours |
| **Total** | **15-21 hours** |

---

## Out of Scope (Future Enhancements)

- Multiple chunking strategies
- Hybrid search (vector + keyword)
- Re-ranking layer
- Query rewriting
- Streaming responses
- Multi-turn conversations
- Additional document formats
- Advanced prompt engineering
- User authentication
- Query history database

---

## Next Immediate Steps

1. **Complete core implementations**:
   - `src/retrieval/vector_store.py`
   - `src/generation/llm_client.py`
   - `src/pipeline/rag_pipeline.py`
   - `src/api/main.py`

2. **Set up KG-RAG dataset**:
   - Clone repository to `data/kg-rag-dataset/`
   - Select 1-2 domains (Finance, Legal recommended)
   - Parse Q&A files

3. **Build test infrastructure**:
   - Create test case converter
   - Implement deepeval metrics
   - Set up pytest configuration

4. **Run baseline evaluation**:
   - Ingest documents
   - Run test suite
   - Document results
   - Identify improvements
