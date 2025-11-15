"""
ChromaDB vector store wrapper for RAG pipeline.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaDBVectorStore:
    """
    ChromaDB vector store wrapper with CRUD operations.
    
    Provides an interface for managing embeddings and metadata with ChromaDB,
    including persistence, batching, and metadata filtering.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection to work with
            persist_directory: Path for persistent storage. If None, uses environment
                             variable CHROMA_PERSIST_DIRECTORY or defaults to ./data/vector_db
            distance_metric: Distance metric for similarity search (cosine, l2, ip)

        Raises:
            ValueError: If distance_metric is not supported
        """
        if distance_metric not in ["cosine", "l2", "ip"]:
            raise ValueError(
                f"Unsupported distance metric: {distance_metric}. "
                "Must be one of: cosine, l2, ip"
            )

        # Set persist directory
        if persist_directory is None:
            persist_directory = os.getenv(
                "CHROMA_PERSIST_DIRECTORY", "./data/vector_db"
            )

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Initialize ChromaDB client with persistence
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(self.settings)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": distance_metric,
                "hnsw:M": 16,
                "hnsw:ef_construction": 200,
            },
        )

        logger.info(
            f"Initialized ChromaDB vector store - "
            f"collection: {collection_name}, "
            f"metric: {distance_metric}, "
            f"persist_dir: {persist_directory}"
        )

    def add(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        ids: List[str],
    ) -> None:
        """
        Add embeddings with metadata and documents to the vector store.

        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries for each embedding
            documents: List of source documents/chunks
            ids: List of unique identifiers for each embedding

        Raises:
            ValueError: If input lists have mismatched lengths
            Exception: If ChromaDB add operation fails
        """
        if not (len(embeddings) == len(metadatas) == len(documents) == len(ids)):
            raise ValueError(
                "Mismatched lengths: embeddings, metadatas, documents, and ids "
                "must all have the same length"
            )

        if len(embeddings) == 0:
            logger.warning("Attempting to add empty batch")
            return

        try:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids,
            )
            logger.info(f"Added {len(embeddings)} embeddings to collection")
        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {e}")
            raise

    def add_batch(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        ids: List[str],
        batch_size: int = 100,
    ) -> None:
        """
        Add embeddings in batches for memory efficiency.

        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: List of source documents
            ids: List of unique identifiers
            batch_size: Number of items per batch

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not (len(embeddings) == len(metadatas) == len(documents) == len(ids)):
            raise ValueError(
                "Mismatched lengths: embeddings, metadatas, documents, and ids "
                "must all have the same length"
            )

        total = len(embeddings)
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)

            batch_embeddings = embeddings[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            self.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents,
                ids=batch_ids,
            )

            logger.info(
                f"Processed batch {batch_idx + 1}/{num_batches} "
                f"({end_idx}/{total} total)"
            )

    def upsert(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        ids: List[str],
    ) -> None:
        """
        Add or update embeddings (upsert operation).

        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: List of source documents
            ids: List of unique identifiers

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not (len(embeddings) == len(metadatas) == len(documents) == len(ids)):
            raise ValueError(
                "Mismatched lengths: embeddings, metadatas, documents, and ids "
                "must all have the same length"
            )

        if len(embeddings) == 0:
            logger.warning("Attempting to upsert empty batch")
            return

        try:
            self.collection.upsert(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids,
            )
            logger.info(f"Upserted {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error upserting embeddings in ChromaDB: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar embeddings.

        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of top results to return
            where: Optional metadata filter for ChromaDB query

        Returns:
            Dictionary containing:
                - ids: List of result IDs
                - distances: List of distances from query
                - metadatas: List of metadata for each result
                - documents: List of document chunks

        Raises:
            ValueError: If top_k is invalid or query_embeddings is empty
            Exception: If ChromaDB query fails
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if len(query_embeddings) == 0:
            raise ValueError("query_embeddings cannot be empty")

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

            logger.info(
                f"Query returned {len(results['ids'][0]) if results['ids'] else 0} results"
            )
            return results

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise

    def delete(self, ids: List[str]) -> None:
        """
        Delete embeddings by ID.

        Args:
            ids: List of embedding IDs to delete

        Raises:
            ValueError: If ids list is empty
            Exception: If ChromaDB delete fails
        """
        if len(ids) == 0:
            logger.warning("Attempting to delete empty list of IDs")
            return

        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings")
        except Exception as e:
            logger.error(f"Error deleting embeddings from ChromaDB: {e}")
            raise

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete embeddings matching metadata filter.

        Args:
            where: Metadata filter condition

        Returns:
            Number of embeddings deleted

        Raises:
            Exception: If ChromaDB delete fails
        """
        try:
            # Query all matching documents
            results = self.collection.get(where=where, include=[])
            ids_to_delete = results["ids"]

            if ids_to_delete:
                self.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} embeddings by metadata")
                return len(ids_to_delete)

            logger.info("No embeddings found matching metadata filter")
            return 0

        except Exception as e:
            logger.error(f"Error deleting by metadata in ChromaDB: {e}")
            raise

    def get_count(self) -> int:
        """
        Get total number of embeddings in collection.

        Returns:
            Total count of embeddings
        """
        try:
            count = self.collection.count()
            logger.info(f"Collection contains {count} embeddings")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            raise

    def get_all(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve all embeddings and metadata from the collection.

        Args:
            include: Optional list of fields to include (ids, embeddings, documents, metadatas)

        Returns:
            Dictionary containing all embeddings and metadata

        Raises:
            Exception: If ChromaDB get operation fails
        """
        if include is None:
            include = ["embeddings", "documents", "metadatas"]

        try:
            results = self.collection.get(include=include)
            logger.info(f"Retrieved all {len(results['ids'])} embeddings")
            return results
        except Exception as e:
            logger.error(f"Error retrieving all embeddings from ChromaDB: {e}")
            raise

    def persist(self) -> None:
        """
        Persist the collection to disk.

        This ensures all data is written to the persist_directory.
        """
        try:
            self.client.persist()
            logger.info("Vector store persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting vector store: {e}")
            raise

    def clear(self) -> None:
        """
        Delete all embeddings from the collection.

        WARNING: This operation is irreversible.
        """
        try:
            # Get all IDs and delete them
            results = self.collection.get(include=[])
            if results["ids"]:
                self.delete(ids=results["ids"])
            logger.warning("Collection cleared - all embeddings deleted")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
