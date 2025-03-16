"""
Document Processing Module for RAG Chatbot
Handles extraction of text from Word documents, Excel files, and PDFs using LangChain document loaders.
"""

import os
import pandas as pd
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
        Process all supported documents in a directory and its subdirectories.
        
        Args:
            directory_path: Path to the directory containing documents
                
        Returns:
            Dictionary mapping file paths to their chunked text content
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        processed_files = {}
        supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls']
        
        # Walk through directory and all subdirectories
        for root, _, files in os.walk(directory_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    # Construct the full file path
                    file_path = os.path.join(root, filename)
                    try:
                        # Get relative path from the base directory to use as key
                        rel_path = os.path.relpath(file_path, directory_path)
                        processed_files[rel_path] = self.process_file(file_path)
                        print(f"Processed {rel_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        return processed_files
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file using LangChain's PyPDFLoader."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load a Word document using LangChain's Docx2txtLoader."""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def _load_excel(self, file_path: str, header_row: int = 0) -> List[Document]:
        """
        Load an Excel file preserving headers and sheet structure.
        Uses pandas to maintain the tabular data structure and headers.
        
        Args:
            file_path: Path to the Excel file
            header_row: Row index (0-based) containing the column headers (default: 0)
            
        Returns:
            List of Document objects with properly formatted Excel content
        """
        import pandas as pd
        from langchain_core.documents import Document
        
        documents = []
        
        # Read all sheets in the Excel file
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        for sheet_name in sheet_names:
            # Read the sheet into a pandas DataFrame with specified header row
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
            
            # Skip empty sheets
            if df.empty:
                continue
            
            # Format the DataFrame as text with headers
            formatted_text = f"# Sheet: {sheet_name}\n\n"
            
            # Add column headers as a section
            formatted_text += "## Headers\n"
            formatted_text += ", ".join(df.columns.astype(str)) + "\n\n"
            
            # Format the data with row indices and header references
            formatted_text += "## Data\n"
            
            # Convert DataFrame to string representation with clear header alignment
            # Use to_string() for better formatting of tabular data
            table_str = df.to_string(index=False)
            formatted_text += table_str + "\n\n"
            
            # Create document with metadata for this sheet
            doc = Document(
                page_content=formatted_text,
                metadata={
                    "source": file_path,
                    "file_type": "excel",
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            )
            documents.append(doc)
            
            # For large sheets, create additional documents with specific data subsets
            # to improve retrieval granularity
            if len(df) > 50:  # Only for larger sheets
                # Process the sheet in smaller chunks with column headers
                chunk_size = 25  # rows per chunk
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    chunk_text = f"# Sheet: {sheet_name} (Rows {i}-{i+len(chunk_df)-1})\n\n"
                    chunk_text += "## Headers\n"
                    chunk_text += ", ".join(df.columns.astype(str)) + "\n\n"
                    chunk_text += "## Data\n"
                    chunk_text += chunk_df.to_string(index=False) + "\n"
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "source": file_path,
                            "file_type": "excel",
                            "sheet_name": sheet_name,
                            "row_range": f"{i}-{i+len(chunk_df)-1}",
                            "row_count": len(chunk_df),
                            "column_count": len(df.columns)
                        }
                    )
                    documents.append(chunk_doc)
        
        return documents
    
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