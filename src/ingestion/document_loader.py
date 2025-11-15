"""
Document loading and preprocessing.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class Document:
    """Represents a document with content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    """Load documents from various sources."""
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file and extract text with page metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects, one per page
        """
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
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all documents from all PDFs
        """
        all_documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            documents = self.load_pdf(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Loaded total of {len(all_documents)} documents")
        return all_documents
