"""
Vector Database Module for RAG Chatbot
Handles document embedding storage and retrieval using Chroma.
"""

import os
from typing import List, Dict, Any, Optional
import uuid

# LangChain components
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class VectorStore:
    """
    A class for managing document embeddings using ChromaDB as the vector database.
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "text-embedding-3-small",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the VectorStore with ChromaDB.
        
        Args:
            persist_directory: Directory to persist the ChromaDB database
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env variable)
        """
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Check if OpenAI API key is available
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
        # Create embeddings instance
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.openai_api_key
        )
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add text chunks to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dictionaries for each text chunk
            
        Returns:
            List of IDs for the added documents
        """
        if not metadatas:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # Generate IDs if not provided in metadata
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add texts to vector store
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Persist the vector store to disk
        self.vector_store.persist()
        
        return ids
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add LangChain Document objects to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of IDs for the added documents
        """
        # Add documents to vector store
        ids = self.vector_store.add_documents(documents)
        
        # Persist the vector store to disk
        self.vector_store.persist()
        
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of Document objects that are most similar to the query
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of tuples containing (Document, score) pairs
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            A retriever that can be used in a RetrievalQA chain
        """
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def clear(self):
        """
        Clear all documents from the vector store.
        """
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.vector_store.persist()


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize vector store
    vector_store = VectorStore(persist_directory="./chroma_db")
    
    # Add texts
    texts = [
        "This is a document about artificial intelligence.",
        "This document discusses machine learning concepts.",
        "Natural language processing is a subfield of AI."
    ]
    
    metadatas = [
        {"source": "ai_doc", "author": "John Doe"},
        {"source": "ml_doc", "author": "Jane Smith"},
        {"source": "nlp_doc", "author": "Bob Johnson"}
    ]
    
    ids = vector_store.add_texts(texts, metadatas)
    print(f"Added {len(ids)} documents to vector store.")
    
    # Perform similarity search
    results = vector_store.similarity_search("What is AI?", k=2)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")
