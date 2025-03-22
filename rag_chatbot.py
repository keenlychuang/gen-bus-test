"""
RAG Chatbot Core Module with conversation history
Integrates document processing, vector database, and OpenAI API for question answering.
"""

import os
import time 
from typing import List, Dict, Any, Optional, Tuple

# LangChain components
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

# Local modules
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.prompt_loader import load_prompt
from utils.query_processor import process_query

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token."""
        self.text += token
        
    def get_text(self):
        return self.text

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
        
        # Initialize OpenAI chat model with streaming enabled
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=self.openai_api_key,
            streaming=True  # Enable streaming
        )
        
        # Create the retriever
        self.retriever = None
        
        # Conversation history
        self.conversation_history = []
        
        # Load prompts from yaml
        self.query_rewriter_prompt = load_prompt("query_rewriter")
        self.qa_prompt = load_prompt("qa_system")
        
        # Initialize the QA chain
        self.qa_chain = None
        self.rewriter_chain = None
    
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
        Create the question-answering chain using LangChain with streaming support.
        """
        # Create the query rewriter chain
        self.rewriter_chain = (
            self.query_rewriter_prompt | self.llm | StrOutputParser()
        )
        
        # Define the RAG pipeline with streaming support
        self.qa_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
                "history": lambda _: self._format_history_for_prompt()
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_history_for_prompt(self) -> str:
        """Format conversation history for inclusion in the prompt."""
        if not self.conversation_history:
            return "No previous conversation."
            
        formatted_history = ""
        for i, (question, answer) in enumerate(self.conversation_history):
            formatted_history += f"Question {i+1}: {question}\n"
            formatted_history += f"Answer {i+1}: {answer}\n\n"
        
        # Limit the history length to avoid exceeding context limits
        # Take only the most recent 3 exchanges if history is long
        if len(self.conversation_history) > 3:
            recent_history = self.conversation_history[-3:]
            formatted_history = "...\n"
            for i, (question, answer) in enumerate(recent_history):
                idx = len(self.conversation_history) - 3 + i + 1
                formatted_history += f"Question {idx}: {question}\n"
                formatted_history += f"Answer {idx}: {answer}\n\n"
        
        return formatted_history.strip()
    
    def _convert_to_langchain_messages(self) -> List:
        """Convert conversation history to LangChain message format for the rewriter."""
        messages = []
        for question, answer in self.conversation_history:
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=answer))
        return messages
    
    # Modify your _rewrite_question method to incorporate keyword processing:
    async def _rewrite_question(self, question: str) -> str:
        """Rewrite contextual questions to standalone questions using conversation history."""
        if not self.conversation_history:
            # No need for contextual processing when there's no history
            return question

        try:
            # First, use existing conversation context handling
            history_messages = self._convert_to_langchain_messages()
            rewritten_question = self.rewriter_chain.invoke({
                "history": history_messages,
                "question": question
            })
            
            # Now enhance with keyword processing
            processed = process_query(rewritten_question)
            
            print(f"Original question: {question}")
            print(f"Rewritten question: {rewritten_question}")
            print(f"Query type: {processed['query_type']}")
            print(f"Keywords: {processed['keywords']}")
            
            # For now, still return the rewritten question
            # We'll use the variations in the retrieval step
            return rewritten_question
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return question
    
    async def ask(self, question: str, streaming=False):
        """
        Ask a question and get an answer based on the loaded documents.
        
        Args:
            question: The question to ask
            streaming: Whether to use streaming
            
        Returns:
            Answer to the question or tuple of (handler, async_generator) for streaming
        """
        if not self.retriever or not self.qa_chain:
            return "Please load documents first using the load_documents method."
        
        try:
            # Rewrite the question if it's a contextual follow-up
            rewritten_question = await self._rewrite_question(question)

            # Process the rewritten query to get variations
            processed = process_query(rewritten_question)
            
            # Get documents using the original query and enhanced keywords
            # This is where we'll later implement the hybrid approach
            # For now, just use the keyword-enriched query if available
            search_query = processed["variations"][1] if len(processed["variations"]) > 1 else rewritten_question
            
            if streaming:
                # Create the callback handler for streaming
                handler = StreamingCallbackHandler()
                # Create config with the handler
                config = RunnableConfig(callbacks=[handler])
                # Return both the handler and the async generator
                generator = self.qa_chain.astream(rewritten_question, config=config)
                return handler, generator
            else:
                # Non-streaming path
                answer = self.qa_chain.invoke(rewritten_question)
                # Add to conversation history
                self.conversation_history.append((question, answer))
                return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def ask_sync(self, question: str, streaming=False):
        """
        Synchronous version of ask method for compatibility.
        """
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if streaming:
                # Not easily doable synchronously
                # Just use non-streaming version
                return loop.run_until_complete(self.ask(question, streaming=False))
            else:
                return loop.run_until_complete(self.ask(question, streaming=False))
        finally:
            loop.close()
    
    def clear_documents(self):
        """
        Clear all documents from the vector store.
        """
        self.vector_store.clear()
        self.retriever = None
        self.qa_chain = None
        print("All documents have been cleared from the vector store")
    
    def clear_history(self):
        """
        Clear conversation history.
        """
        self.conversation_history = []
        print("Conversation history has been cleared")


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize RAG chatbot
    chatbot = RAGChatbot()
    
    # Load documents
    chatbot.load_documents(file_paths=["path/to/document.pdf"])
    
    # Ask questions
    answer1 = chatbot.ask_sync("What is the main topic of the document?")
    print(f"Question: What is the main topic of the document?")
    print(f"Answer: {answer1}")
    
    answer2 = chatbot.ask_sync("Can you tell me more about it?")
    print(f"Question: Can you tell me more about it?")
    print(f"Answer: {answer2}")