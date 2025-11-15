"""
Integration tests for ChromaDB vector store with realistic scenarios.
"""
import os
import shutil
import tempfile

import pytest

from src.retrieval.vector_store import ChromaDBVectorStore


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_integration_test_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_persist_dir):
    """Create a ChromaDB vector store instance for testing."""
    store = ChromaDBVectorStore(
        collection_name="integration_test_collection",
        persist_directory=temp_persist_dir,
        distance_metric="cosine",
    )
    yield store
    store.clear()


class TestIntegration:
    """Integration tests for realistic RAG scenarios."""

    def test_end_to_end_document_ingestion_and_retrieval(self, vector_store):
        """Test complete workflow: ingest documents and retrieve similar chunks."""
        # Simulate document ingestion from multiple sources
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing deals with text and speech.",
            "Transformers have revolutionized NLP in recent years.",
        ]

        # Create simple embeddings (in real scenario, these come from embedding model)
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # ML/AI
            [0.15, 0.25, 0.35, 0.45],  # Neural nets
            [0.12, 0.22, 0.32, 0.42],  # Deep learning
            [0.2, 0.3, 0.4, 0.5],  # NLP
            [0.22, 0.32, 0.42, 0.52],  # Transformers
        ]

        metadatas = [
            {"source": "doc1.pdf", "page": 1, "chunk_id": 0},
            {"source": "doc1.pdf", "page": 2, "chunk_id": 1},
            {"source": "doc1.pdf", "page": 3, "chunk_id": 2},
            {"source": "doc2.pdf", "page": 1, "chunk_id": 0},
            {"source": "doc2.pdf", "page": 2, "chunk_id": 1},
        ]

        ids = [f"chunk_{i}" for i in range(5)]

        # Ingest all documents
        vector_store.add_batch(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
            batch_size=2,
        )

        # Verify ingestion
        assert vector_store.get_count() == 5

        # Query for similar content
        query_embedding = [[0.11, 0.21, 0.31, 0.41]]  # Similar to first two chunks
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=2,
        )

        assert len(results["ids"][0]) == 2
        assert all(isinstance(id, str) for id in results["ids"][0])
        assert all(isinstance(doc, str) for doc in results["documents"][0])

    def test_multi_source_document_management(self, vector_store):
        """Test managing embeddings from multiple documents with metadata filtering."""
        # Simulate ingesting from 3 different documents
        doc_sources = ["financial_report.pdf", "research_paper.pdf", "blog_post.md"]
        num_chunks_per_doc = 3

        all_embeddings = []
        all_metadatas = []
        all_documents = []
        all_ids = []

        for doc_idx, source in enumerate(doc_sources):
            for chunk_idx in range(num_chunks_per_doc):
                # Create distinct embeddings for each chunk
                base_val = (doc_idx * 0.1) + (chunk_idx * 0.01)
                embedding = [
                    base_val,
                    base_val + 0.1,
                    base_val + 0.2,
                    base_val + 0.3,
                ]

                all_embeddings.append(embedding)
                all_metadatas.append({
                    "source": source,
                    "chunk_index": chunk_idx,
                    "document_index": doc_idx,
                })
                all_documents.append(
                    f"Content from {source} chunk {chunk_idx}"
                )
                all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")

        # Ingest all documents
        vector_store.add_batch(
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=all_documents,
            ids=all_ids,
        )

        assert vector_store.get_count() == 9

        # Query within specific document
        query_embedding = [[0.0, 0.1, 0.2, 0.3]]
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=5,
            where={"source": "financial_report.pdf"},
        )

        # Should only get results from financial_report.pdf
        for id_str in results["ids"][0]:
            assert "doc0_chunk" in id_str

        # Delete chunks from one document
        deleted_count = vector_store.delete_by_metadata(
            where={"source": "research_paper.pdf"}
        )
        assert deleted_count == 3
        assert vector_store.get_count() == 6

    def test_incremental_updates_and_versioning(self, vector_store):
        """Test updating documents and maintaining version history."""
        # Initial version of chunks
        initial_ids = ["chunk_1", "chunk_2"]
        initial_embeddings = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
        initial_metadatas = [
            {"document": "doc1", "version": 1},
            {"document": "doc1", "version": 1},
        ]
        initial_documents = ["Original content 1", "Original content 2"]

        vector_store.add(
            embeddings=initial_embeddings,
            metadatas=initial_metadatas,
            documents=initial_documents,
            ids=initial_ids,
        )

        assert vector_store.get_count() == 2

        # Add new version with upsert
        updated_embeddings = [[0.15, 0.25, 0.35, 0.45], [0.25, 0.35, 0.45, 0.55]]
        updated_metadatas = [
            {"document": "doc1", "version": 2},
            {"document": "doc1", "version": 2},
        ]
        updated_documents = ["Updated content 1", "Updated content 2"]

        vector_store.upsert(
            embeddings=updated_embeddings,
            metadatas=updated_metadatas,
            documents=updated_documents,
            ids=initial_ids,
        )

        # Should still be 2 embeddings (updated, not added)
        assert vector_store.get_count() == 2

        # Verify content was updated
        all_data = vector_store.get_all(include=["documents", "metadatas"])
        versions = [m["version"] for m in all_data["metadatas"]]
        assert all(v == 2 for v in versions)

    def test_large_batch_ingestion(self, vector_store):
        """Test handling large batches of documents efficiently."""
        num_documents = 500
        batch_size = 50

        embeddings = [
            [float(i % 256) / 256, float((i + 1) % 256) / 256,
             float((i + 2) % 256) / 256, float((i + 3) % 256) / 256]
            for i in range(num_documents)
        ]
        metadatas = [
            {"index": i, "batch": i // batch_size}
            for i in range(num_documents)
        ]
        documents = [f"Document content {i}" for i in range(num_documents)]
        ids = [f"doc_{i}" for i in range(num_documents)]

        # Ingest with batching
        vector_store.add_batch(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
            batch_size=batch_size,
        )

        assert vector_store.get_count() == num_documents

        # Query should still work efficiently
        query_embedding = [[0.5, 0.5, 0.5, 0.5]]
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=10,
        )

        assert len(results["ids"][0]) == 10

    def test_persistence_across_instances(self, temp_persist_dir):
        """Test that data persists across different vector store instances."""
        # Create first instance and add data
        store1 = ChromaDBVectorStore(
            collection_name="persistent_collection",
            persist_directory=temp_persist_dir,
        )

        store1.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"test": "data"}],
            documents=["Test content"],
            ids=["persistent_id"],
        )

        assert store1.get_count() == 1
        store1.persist()

        # Create second instance with same persist directory
        store2 = ChromaDBVectorStore(
            collection_name="persistent_collection",
            persist_directory=temp_persist_dir,
        )

        # Data should be available
        assert store2.get_count() == 1

        # Verify content is the same
        all_data = store2.get_all(include=["ids", "documents"])
        assert all_data["ids"] == ["persistent_id"]
        assert all_data["documents"][0] == "Test content"

    def test_complex_metadata_filtering_scenarios(self, vector_store):
        """Test various metadata filtering scenarios."""
        # Ingest documents with rich metadata
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5],
            [0.25, 0.35, 0.45, 0.55],
        ]

        metadatas = [
            {"source": "doc1", "category": "tech", "year": 2023},
            {"source": "doc1", "category": "finance", "year": 2023},
            {"source": "doc2", "category": "tech", "year": 2024},
            {"source": "doc2", "category": "finance", "year": 2024},
        ]

        documents = [
            "Tech content from doc1",
            "Finance content from doc1",
            "Tech content from doc2",
            "Finance content from doc2",
        ]

        ids = ["doc1_tech", "doc1_fin", "doc2_tech", "doc2_fin"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        # Query with single field filter
        query_embedding = [[0.1, 0.2, 0.3, 0.4]]
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=10,
            where={"category": "tech"},
        )
        returned_ids = results["ids"][0]
        assert all("tech" in id for id in returned_ids)

        # Delete by category
        deleted = vector_store.delete_by_metadata(where={"category": "finance"})
        assert deleted == 2
        assert vector_store.get_count() == 2

    def test_concurrent_read_after_write(self, vector_store):
        """Test that data is immediately readable after writing."""
        # Add batch of documents
        embeddings = [
            [float(i) / 10, float(i + 1) / 10, float(i + 2) / 10, float(i + 3) / 10]
            for i in range(10)
        ]
        metadatas = [{"index": i} for i in range(10)]
        documents = [f"Content {i}" for i in range(10)]
        ids = [f"id_{i}" for i in range(10)]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        # Immediately query
        query_embedding = [[0.0, 0.1, 0.2, 0.3]]
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=5,
        )

        # Should get results immediately
        assert len(results["ids"][0]) == 5
        assert all(id in ids for id in results["ids"][0])
