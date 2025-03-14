"""
Document Processing Module for RAG Chatbot
Handles extraction of text from Word documents, Excel files, and PDFs using LangChain document loaders.
"""

import os
from typing import List, Dict, Any, Optional

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_core.documents import Document


class DocumentProcessor:
    """
    A class for processing different document types (PDF, Word, Excel) and extracting text content
    using LangChain document loaders.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentProcessor with text chunking parameters.
        
        Args:
            chunk_size: The size of text chunks for splitting documents
            chunk_overlap: The overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a file based on its extension and return chunked text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of text chunks extracted from the document
        
        Raises:
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Load documents using LangChain document loaders
        try:
            if file_extension == '.pdf':
                documents = self._load_pdf(file_path)
            elif file_extension == '.docx':
                documents = self._load_docx(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                documents = self._load_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Extract text from documents
            text = self._extract_text_from_documents(documents)
            
            # Split the text into chunks
            return self.text_splitter.split_text(text)
            
        except Exception as e:
            raise Exception(f"Error processing {file_path}: {str(e)}")
    
    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            Dictionary mapping file names to their chunked text content
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        processed_files = {}
        supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls']
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    try:
                        processed_files[filename] = self.process_file(file_path)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
        
        return processed_files
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file using LangChain's PyPDFLoader."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load a Word document using LangChain's Docx2txtLoader."""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def _load_excel(self, file_path: str) -> List[Document]:
        """Load an Excel file using LangChain's UnstructuredExcelLoader."""
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        return loader.load()
    
    def _extract_text_from_documents(self, documents: List[Document]) -> str:
        """Extract text from a list of LangChain Document objects."""
        text = ""
        for doc in documents:
            text += doc.page_content + "\n\n"
        return text


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Example: chunks = processor.process_file("path/to/document.pdf")