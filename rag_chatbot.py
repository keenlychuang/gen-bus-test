"""
RAG Chatbot Core Module with source citations
Integrates document processing, vector database, and OpenAI API for question answering.
"""

import os
from typing import List, Dict, Any, Optional

# LangChain components
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Local modules
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore


class RAGChatbot:
    """
    A Retrieval-Augmented Generation (RAG) chatbot that uses OpenAI API
    to answer questions based on the content of processed documents.
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini-2024-07-18",
                 temperature: float = 0.0,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG chatbot.
        
        Args:
            persist_directory: Directory to persist the ChromaDB database
            openai_api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env variable)
            model_name: OpenAI model to use for chat completion
            temperature: Temperature parameter for chat completion
            chunk_size: Size of text chunks for document processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Check if OpenAI API key is available
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=self.openai_api_key
        )
        
        # Create the retriever
        self.retriever = None
        
        # Define the QA prompt template with citation instructions
        self.qa_prompt = PromptTemplate.from_template(
            """You are a expert research assistant that answers questions based on the provided context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer the question based only on the provided context. If the context doesn't contain 
            the information needed to answer the question, say "I don't have enough information to 
            answer this question." and suggest what other information might be helpful.
            
            Important: For each piece of information you use, include a citation to the source document 
            using footnote notation [1], [2], etc. At the end of your answer, list all the source 
            documents you cited with detailed information:

            1. For PDF or Word documents, use the filename:
               [1] filename.pdf
               [2] another_document.docx
            
            2. For Excel files, include the sheet name and row information:
               [3] spreadsheet.xlsx (Sheet: Sales, Rows: 10-35)
               [4] data.xlsx (Sheet: Q2 Report)
            
            Use all available metadata to make the citation as specific as possible, including 
            "sheet_name", "row_range", and other Excel-specific metadata when available.
            
            Answer:"""
        )
        
        # Initialize the QA chain
        self.qa_chain = None
    
    def load_documents(self, file_paths: List[str] = None, directory_path: str = None):
        """
        Load and process documents from files or a directory.
        
        Args:
            file_paths: List of file paths to process
            directory_path: Directory path containing documents to process
                
        Returns:
            Number of documents loaded
        """
        processed_chunks = []
        metadatas = []
        
        # Process individual files
        if file_paths:
            for file_path in file_paths:
                try:
                    chunks = self.document_processor.process_file(file_path)
                    file_name = os.path.basename(file_path)
                    
                    # Create metadata for each chunk
                    file_metadatas = [{"source": file_name, "file_path": file_path} for _ in chunks]
                    
                    processed_chunks.extend(chunks)
                    metadatas.extend(file_metadatas)
                    
                    print(f"Processed {file_name}: {len(chunks)} chunks extracted")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        # Process all files in directory and subdirectories
        if directory_path:
            try:
                processed_files = self.document_processor.process_directory(directory_path)
                
                for rel_path, chunks in processed_files.items():
                    # Construct full file path
                    file_path = os.path.join(directory_path, rel_path)
                    
                    # Create metadata for each chunk
                    file_metadatas = [{"source": rel_path, "file_path": file_path} for _ in chunks]
                    
                    processed_chunks.extend(chunks)
                    metadatas.extend(file_metadatas)
                    
                    print(f"Processed {rel_path}: {len(chunks)} chunks extracted")
            except Exception as e:
                print(f"Error processing directory {directory_path}: {str(e)}")
        
        # Add processed chunks to vector store
        if processed_chunks:
            self.vector_store.add_texts(processed_chunks, metadatas)
            print(f"Added {len(processed_chunks)} chunks to vector store")
            
            # Create retriever
            self.retriever = self.vector_store.get_retriever()
            
            # Create QA chain
            self._create_qa_chain()
        
        return len(processed_chunks)
    
    def _create_qa_chain(self):
        """
        Create the question-answering chain using LangChain.
        """
        # Define the RAG pipeline
        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer based on the loaded documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Answer to the question
        """
        if not self.retriever or not self.qa_chain:
            return "Please load documents first using the load_documents method."
        
        try:
            # Get answer using the QA chain
            answer = self.qa_chain.invoke(question)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def clear_documents(self):
        """
        Clear all documents from the vector store.
        """
        self.vector_store.clear()
        self.retriever = None
        self.qa_chain = None
        print("All documents have been cleared from the vector store")


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize RAG chatbot
    chatbot = RAGChatbot()
    
    # Load documents
    chatbot.load_documents(file_paths=["path/to/document.pdf"])
    
    # Ask a question
    question = "What is the main topic of the document?"
    answer = chatbot.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")