"""
Test script for the RAG Chatbot core
"""

import os
from rag_chatbot import RAGChatbot

def test_rag_chatbot():
    """Test the RAGChatbot class functionality"""
    print("Testing RAGChatbot functionality")
    print("Note: This test requires an OpenAI API key to be set")
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nWARNING: OpenAI API key not found in environment variables")
        print("To run this test, set your API key with:")
        print("export OPENAI_API_KEY='your-api-key'")
        print("or provide it directly in the RAGChatbot initialization")
        return
    
    # Create a test directory for the vector store
    test_dir = "./test_rag_db"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a sample directory for test files if it doesn't exist
    sample_dir = "./data/sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a simple text file to simulate a document
    sample_file = os.path.join(sample_dir, "sample_content.txt")
    with open(sample_file, "w") as f:
        f.write("""
        # Artificial Intelligence Overview
        
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.
        
        ## Machine Learning
        
        Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience.
        
        ## Natural Language Processing
        
        Natural Language Processing (NLP) is a field of AI that gives computers the ability to understand text and spoken words.
        
        ## Computer Vision
        
        Computer Vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.
        """)
    
    try:
        # Initialize RAG chatbot
        print("\nInitializing RAG chatbot...")
        chatbot = RAGChatbot(persist_directory=test_dir)
        print("RAG chatbot initialized successfully")
        
        # Load the sample document
        print("\nLoading sample document...")
        num_chunks = chatbot.load_documents(file_paths=[sample_file])
        print(f"Loaded {num_chunks} chunks from sample document")
        
        # Test asking questions
        test_questions = [
            "What is artificial intelligence?",
            "What are the subfields of AI mentioned in the document?",
            "What is machine learning?",
            "What is not covered in the document?"
        ]
        
        print("\nTesting question answering:")
        for question in test_questions:
            print(f"\nQuestion: {question}")
            answer = chatbot.ask(question)
            print(f"Answer: {answer}")
        
        # Test clearing documents
        print("\nClearing documents from vector store...")
        chatbot.clear_documents()
        
        print("\nRAG chatbot test completed successfully")
        
    except Exception as e:
        print(f"\nError testing RAG chatbot: {str(e)}")
    
    print("\nTo use RAGChatbot in your application:")
    print("1. Initialize: chatbot = RAGChatbot()")
    print("2. Load documents: chatbot.load_documents(file_paths=['document.pdf'])")
    print("3. Ask questions: answer = chatbot.ask('What is in the document?')")

if __name__ == "__main__":
    test_rag_chatbot()
