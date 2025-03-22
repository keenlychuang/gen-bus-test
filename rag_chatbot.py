"""
RAG Chatbot Core Module with conversation history
Integrates document processing, vector database, and OpenAI API for question answering.
"""

import os
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

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, streaming_callback):
        self.streaming_callback = streaming_callback
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token."""
        if self.streaming_callback:
            self.streaming_callback(token)

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
        
        # Define the query rewriter for contextual questions
        self.query_rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a expert research assistant that rewrites ambiguous or contextual follow-up questions into standalone questions that can be understood without conversation history.
            Use the conversation history to understand what the user is referring to, and rewrite their question to be self-contained.
            If the question is already self-contained and clear, return it unchanged.
            
            Examples:
            - "What is its capital?" → "What is the capital of France?" (if previous question was about France)
            - "How many does it have?" → "How many provinces does Canada have?" (if previous question was about Canada)
            - "When was it founded?" → "When was Microsoft founded?" (if previous question was about Microsoft)
            - "Where is Mount Everest located?" → "Where is Mount Everest located?" (already clear, no change needed)
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Rewrite this question to be a standalone question: {question}")
        ])
        
        # Define the QA prompt template
        self.qa_prompt = PromptTemplate.from_template(
            """You are a expert research assistant that answers questions based on the provided context and conversation history.
            
            Context:
            {context}
            
            Conversation History:
            {history}
            
            Question:
            {question}
            
            Answer the question based on the provided context and conversation history. If the context doesn't contain 
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
    
    async def _rewrite_question(self, question: str) -> str:
        """Rewrite contextual questions to standalone questions using conversation history."""
        if not self.conversation_history:
            return question  # No history to use for rewriting
        
        try:
            history_messages = self._convert_to_langchain_messages()
            rewritten_question = self.rewriter_chain.invoke({
                "history": history_messages,
                "question": question
            })
            
            print(f"Original question: {question}")
            print(f"Rewritten question: {rewritten_question}")
            
            return rewritten_question
        except Exception as e:
            print(f"Error rewriting question: {str(e)}")
            return question  # Fall back to original question
    
    async def ask(self, question: str, streaming_callback=None):
        """
        Ask a question and get an answer based on the loaded documents.
        
        Args:
            question: The question to ask
            streaming_callback: Optional callback function for streaming responses
            
        Returns:
            Answer to the question
        """
        if not self.retriever or not self.qa_chain:
            return "Please load documents first using the load_documents method."
        
        try:
            # Rewrite the question if it's a contextual follow-up
            rewritten_question = await self._rewrite_question(question)
            
            # Handle streaming if callback is provided
            if streaming_callback:
                callback_handler = StreamingCallbackHandler(streaming_callback)
                config = RunnableConfig(callbacks=[callback_handler])
                
                answer = ""
                async for chunk in self.qa_chain.astream(rewritten_question, config=config):
                    answer += chunk
            else:
                # Get answer using the QA chain (non-streaming)
                answer = self.qa_chain.invoke(rewritten_question)
            
            # Add to conversation history
            self.conversation_history.append((question, answer))
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    # Update the synchronous version as well
    def ask_sync(self, question: str, streaming_callback=None):
        """
        Synchronous version of ask method for compatibility.
        """
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ask(question, streaming_callback))
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