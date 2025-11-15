"""
Text chunking strategies.
"""
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunk:
    """Represents a text chunk with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: str):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
    
    def __repr__(self) -> str:
        return f"TextChunk(id={self.chunk_id}, length={len(self.text)})"


class DocumentChunker:
    """Chunk documents into smaller pieces for retrieval."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51):
        """
        Initialize chunker with recursive character text splitting.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(
            f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_documents(self, documents: List[Any]) -> List[TextChunk]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of TextChunk objects
        """
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
        """Log statistics about chunk sizes."""
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
