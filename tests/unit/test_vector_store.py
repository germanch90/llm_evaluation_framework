"""
Unit tests for ChromaDB vector store wrapper.
"""
import os
import shutil
import tempfile

import pytest

from src.retrieval.vector_store import ChromaDBVectorStore


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_persist_dir):
    """Create a ChromaDB vector store instance for testing."""
    store = ChromaDBVectorStore(
        collection_name="test_collection",
        persist_directory=temp_persist_dir,
        distance_metric="cosine",
    )
    yield store
    store.clear()


class TestInitialization:
    """Test ChromaDB vector store initialization."""

    def test_init_with_defaults(self, temp_persist_dir):
        """Test initialization with default parameters."""
        store = ChromaDBVectorStore(persist_directory=temp_persist_dir)
        assert store.collection_name == "rag_documents"
        assert store.distance_metric == "cosine"
        assert store.persist_directory == temp_persist_dir
        assert os.path.exists(temp_persist_dir)

    def test_init_with_custom_params(self, temp_persist_dir):
        """Test initialization with custom parameters."""
        store = ChromaDBVectorStore(
            collection_name="custom_collection",
            persist_directory=temp_persist_dir,
            distance_metric="l2",
        )
        assert store.collection_name == "custom_collection"
        assert store.distance_metric == "l2"

    def test_init_invalid_distance_metric(self, temp_persist_dir):
        """Test initialization with invalid distance metric."""
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            ChromaDBVectorStore(
                persist_directory=temp_persist_dir,
                distance_metric="invalid",
            )

    def test_init_creates_persist_dir(self):
        """Test that initialization creates persist directory."""
        temp_dir = tempfile.mkdtemp()
        persist_path = os.path.join(temp_dir, "new_vector_db")
        assert not os.path.exists(persist_path)

        ChromaDBVectorStore(persist_directory=persist_path)
        assert os.path.exists(persist_path)

        # Cleanup
        shutil.rmtree(temp_dir)


class TestAdd:
    """Test add operation."""

    def test_add_single_embedding(self, vector_store):
        """Test adding a single embedding."""
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        metadatas = [{"source": "doc1", "page": 1}]
        documents = ["This is test content"]
        ids = ["id1"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        count = vector_store.get_count()
        assert count == 1

    def test_add_multiple_embeddings(self, vector_store):
        """Test adding multiple embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ]
        metadatas = [
            {"source": "doc1", "page": 1},
            {"source": "doc1", "page": 2},
            {"source": "doc2", "page": 1},
        ]
        documents = [
            "Content 1",
            "Content 2",
            "Content 3",
        ]
        ids = ["id1", "id2", "id3"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        count = vector_store.get_count()
        assert count == 3

    def test_add_mismatched_lengths(self, vector_store):
        """Test add with mismatched input lengths."""
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
        metadatas = [{"source": "doc1"}]  # Only 1 metadata
        documents = ["Content 1", "Content 2"]
        ids = ["id1", "id2"]

        with pytest.raises(ValueError, match="Mismatched lengths"):
            vector_store.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids,
            )

    def test_add_empty_batch(self, vector_store):
        """Test add with empty batch."""
        vector_store.add(
            embeddings=[],
            metadatas=[],
            documents=[],
            ids=[],
        )
        count = vector_store.get_count()
        assert count == 0


class TestAddBatch:
    """Test batch add operation."""

    def test_add_batch_with_single_batch(self, vector_store):
        """Test batch add that fits in single batch."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [
            {"source": "doc1"},
            {"source": "doc2"},
        ]
        documents = ["Content 1", "Content 2"]
        ids = ["id1", "id2"]

        vector_store.add_batch(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
            batch_size=10,
        )

        count = vector_store.get_count()
        assert count == 2

    def test_add_batch_with_multiple_batches(self, vector_store):
        """Test batch add with multiple batches."""
        num_items = 250
        embeddings = [[float(i) / 100 for _ in range(4)] for i in range(num_items)]
        metadatas = [{"index": i} for i in range(num_items)]
        documents = [f"Content {i}" for i in range(num_items)]
        ids = [f"id{i}" for i in range(num_items)]

        vector_store.add_batch(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
            batch_size=100,
        )

        count = vector_store.get_count()
        assert count == num_items

    def test_add_batch_mismatched_lengths(self, vector_store):
        """Test batch add with mismatched lengths."""
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
        metadatas = [{"source": "doc1"}]  # Mismatched
        documents = ["Content 1", "Content 2"]
        ids = ["id1", "id2"]

        with pytest.raises(ValueError, match="Mismatched lengths"):
            vector_store.add_batch(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids,
            )


class TestUpsert:
    """Test upsert operation."""

    def test_upsert_new_embeddings(self, vector_store):
        """Test upserting new embeddings."""
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        metadatas = [{"source": "doc1"}]
        documents = ["Content 1"]
        ids = ["id1"]

        vector_store.upsert(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        count = vector_store.get_count()
        assert count == 1

    def test_upsert_existing_embeddings(self, vector_store):
        """Test upserting existing embeddings (update)."""
        # Add initial embedding
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1", "version": 1}],
            documents=["Original content"],
            ids=["id1"],
        )

        # Upsert with same ID but different content
        vector_store.upsert(
            embeddings=[[0.5, 0.6, 0.7, 0.8]],
            metadatas=[{"source": "doc1", "version": 2}],
            documents=["Updated content"],
            ids=["id1"],
        )

        count = vector_store.get_count()
        assert count == 1

        # Verify content was updated
        results = vector_store.get_all(include=["documents"])
        assert results["documents"][0] == "Updated content"


class TestQuery:
    """Test query operation."""

    def test_query_empty_collection(self, vector_store):
        """Test querying empty collection returns empty results."""
        query_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        
        results = vector_store.query(query_embeddings=query_embeddings, top_k=5)
        # ChromaDB returns empty results for empty collections, not an exception
        assert len(results["ids"][0]) == 0

    def test_query_single_result(self, vector_store):
        """Test query returning single result."""
        # Add embeddings
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        # Query
        query_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        results = vector_store.query(query_embeddings=query_embeddings, top_k=5)

        assert len(results["ids"][0]) > 0
        assert results["ids"][0][0] == "id1"

    def test_query_multiple_results(self, vector_store):
        """Test query returning multiple results."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc3"},
        ]
        documents = ["Content 1", "Content 2", "Content 3"]
        ids = ["id1", "id2", "id3"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        query_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        results = vector_store.query(query_embeddings=query_embeddings, top_k=2)

        assert len(results["ids"][0]) == 2

    def test_query_with_top_k(self, vector_store):
        """Test query respects top_k parameter."""
        embeddings = [
            [float(i) / 100, float(i + 1) / 100, float(i + 2) / 100, float(i + 3) / 100]
            for i in range(10)
        ]
        metadatas = [{"index": i} for i in range(10)]
        documents = [f"Content {i}" for i in range(10)]
        ids = [f"id{i}" for i in range(10)]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        query_embeddings = [[0.0, 0.01, 0.02, 0.03]]
        results = vector_store.query(query_embeddings=query_embeddings, top_k=3)

        assert len(results["ids"][0]) == 3

    def test_query_with_metadata_filter(self, vector_store):
        """Test query with metadata filtering."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [
            {"source": "doc1", "category": "A"},
            {"source": "doc2", "category": "B"},
            {"source": "doc3", "category": "A"},
        ]
        documents = ["Content 1", "Content 2", "Content 3"]
        ids = ["id1", "id2", "id3"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        query_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        results = vector_store.query(
            query_embeddings=query_embeddings,
            top_k=10,
            where={"category": "A"},
        )

        # Should only return results from category A
        returned_ids = results["ids"][0]
        assert all(id in ["id1", "id3"] for id in returned_ids)

    def test_query_invalid_top_k(self, vector_store):
        """Test query with invalid top_k."""
        query_embeddings = [[0.1, 0.2, 0.3, 0.4]]

        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            vector_store.query(query_embeddings=query_embeddings, top_k=0)

    def test_query_empty_embeddings(self, vector_store):
        """Test query with empty embeddings."""
        with pytest.raises(ValueError, match="query_embeddings cannot be empty"):
            vector_store.query(query_embeddings=[], top_k=5)


class TestDelete:
    """Test delete operation."""

    def test_delete_single_embedding(self, vector_store):
        """Test deleting a single embedding."""
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        assert vector_store.get_count() == 1

        vector_store.delete(ids=["id1"])
        assert vector_store.get_count() == 0

    def test_delete_multiple_embeddings(self, vector_store):
        """Test deleting multiple embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5],
        ]
        ids = ["id1", "id2", "id3"]
        metadatas = [{"index": i} for i in range(3)]
        documents = [f"Content {i}" for i in range(3)]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        assert vector_store.get_count() == 3

        vector_store.delete(ids=["id1", "id2"])
        assert vector_store.get_count() == 1

    def test_delete_empty_ids(self, vector_store):
        """Test delete with empty IDs list."""
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        vector_store.delete(ids=[])
        assert vector_store.get_count() == 1


class TestDeleteByMetadata:
    """Test delete by metadata operation."""

    def test_delete_by_metadata(self, vector_store):
        """Test deleting embeddings by metadata filter."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [
            {"source": "doc1", "category": "A"},
            {"source": "doc2", "category": "B"},
            {"source": "doc3", "category": "A"},
        ]
        documents = ["Content 1", "Content 2", "Content 3"]
        ids = ["id1", "id2", "id3"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        assert vector_store.get_count() == 3

        deleted_count = vector_store.delete_by_metadata(where={"category": "A"})
        assert deleted_count == 2
        assert vector_store.get_count() == 1

    def test_delete_by_metadata_no_matches(self, vector_store):
        """Test delete by metadata with no matches."""
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1", "category": "A"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        deleted_count = vector_store.delete_by_metadata(where={"category": "B"})
        assert deleted_count == 0
        assert vector_store.get_count() == 1


class TestUtility:
    """Test utility methods."""

    def test_get_count(self, vector_store):
        """Test getting collection count."""
        assert vector_store.get_count() == 0

        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        assert vector_store.get_count() == 1

    def test_get_all(self, vector_store):
        """Test retrieving all embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [
            {"source": "doc1"},
            {"source": "doc2"},
        ]
        documents = ["Content 1", "Content 2"]
        ids = ["id1", "id2"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        results = vector_store.get_all(
            include=["ids", "documents", "metadatas", "embeddings"]
        )

        assert len(results["ids"]) == 2
        assert len(results["documents"]) == 2
        assert len(results["metadatas"]) == 2
        assert len(results["embeddings"]) == 2

    def test_persist(self, vector_store, temp_persist_dir):
        """Test persistence to disk."""
        vector_store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "doc1"}],
            documents=["Content 1"],
            ids=["id1"],
        )

        vector_store.persist()

        # Verify files were created
        assert os.path.exists(temp_persist_dir)
        # ChromaDB creates several files in the persist directory
        assert len(os.listdir(temp_persist_dir)) > 0

    def test_clear(self, vector_store):
        """Test clearing collection."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ]
        metadatas = [{"index": i} for i in range(2)]
        documents = ["Content 1", "Content 2"]
        ids = ["id1", "id2"]

        vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

        assert vector_store.get_count() == 2

        vector_store.clear()
        assert vector_store.get_count() == 0
