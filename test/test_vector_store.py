"""
Test script for the VectorStore module
"""

import os
from utils.vector_store import VectorStore
from langchain_core.documents import Document

def test_vector_store():
    """Test the VectorStore class functionality"""
    print("Testing VectorStore functionality")
    print("Note: This test requires an OpenAI API key to be set")
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nWARNING: OpenAI API key not found in environment variables")
        print("To run this test, set your API key with:")
        print("export OPENAI_API_KEY='your-api-key'")
        print("or provide it directly in the VectorStore initialization")
        return
    
    # Create a test directory for the vector store
    test_dir = "./test_chroma_db"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Initialize vector store
        vector_store = VectorStore(persist_directory=test_dir)
        print("\nVectorStore initialized successfully")
        
        # Test adding texts
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
        
        print("\nAdding sample texts to vector store...")
        ids = vector_store.add_texts(texts, metadatas)
        print(f"Added {len(ids)} documents to vector store")
        
        # Test adding Document objects
        documents = [
            Document(page_content="Deep learning is a subset of machine learning.", 
                     metadata={"source": "deep_learning_doc", "author": "Alice Brown"}),
            Document(page_content="Computer vision is the field of AI that enables computers to see.",
                     metadata={"source": "cv_doc", "author": "Charlie Davis"})
        ]
        
        print("\nAdding Document objects to vector store...")
        doc_ids = vector_store.add_documents(documents)
        print(f"Added {len(doc_ids)} Document objects to vector store")
        
        # Test similarity search
        print("\nPerforming similarity search for 'What is AI?'")
        results = vector_store.similarity_search("What is AI?", k=2)
        
        print("\nSearch results:")
        for i, doc in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("---")
        
        # Test similarity search with scores
        print("\nPerforming similarity search with scores for 'machine learning'")
        results_with_scores = vector_store.similarity_search_with_score("machine learning", k=2)
        
        print("\nSearch results with scores:")
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print(f"Relevance Score: {score}")
            print("---")
        
        # Test retriever
        print("\nGetting retriever from vector store")
        retriever = vector_store.get_retriever(search_kwargs={"k": 3})
        print("Retriever created successfully")
        
        print("\nVector store test completed successfully")
        
    except Exception as e:
        print(f"\nError testing vector store: {str(e)}")
    
    print("\nTo use VectorStore in your application:")
    print("1. Initialize: vector_store = VectorStore(persist_directory='./chroma_db')")
    print("2. Add documents: vector_store.add_texts(texts, metadatas)")
    print("3. Search: results = vector_store.similarity_search(query)")
    print("4. Get retriever: retriever = vector_store.get_retriever()")

if __name__ == "__main__":
    test_vector_store()
