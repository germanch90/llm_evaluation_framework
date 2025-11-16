"""
Embedding generation using Ollama.
"""
import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_host: str | None = None,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Name of the Ollama embedding model
            ollama_host: URL of Ollama server (defaults to OLLAMA_HOST env var)
        """
        self.model_name = model_name
        resolved_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_host = resolved_host.rstrip("/")
        self.api_url = f"{self.ollama_host}/api/embeddings"

        logger.info("Initialized embedding model: %s (host=%s)", model_name, self.ollama_host)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
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
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return embeddings
