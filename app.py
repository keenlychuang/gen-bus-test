"""
Streamlit UI for RAG Chatbot
Provides a user-friendly interface with custom styling.
"""

import os
import tempfile
import streamlit as st
from rag_chatbot import RAGChatbot

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# CSS for styling
def get_css():
    return """
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #F5F5DC; /* Beige background */
            color: #000000; /* Black text */
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-12oz5g7 {
            background-color: #D2B48C; /* Tan/taupe sidebar */
        }
        
        /* Chat message containers */
        .stChatMessage {
            background-color: #FFF8DC; /* Cream color for messages */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* User messages */
        .stChatMessage[data-testid="user-message"] {
            background-color: #E8DFD8; /* Light taupe for user messages */
        }
        
        /* Assistant messages */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #F9F6F0; /* Lighter cream for assistant */
        }
        
        /* Input fields */
        .stTextInput input, .stFileUploader label {
            background-color: #FFF8E7; /* Very light cream */
            border: 1px solid #D2B48C; /* Taupe border */
        }
        
        /* Buttons */
        .stButton button {
            background-color: #C8B39C; /* Muted taupe */
            color: black;
            border: none;
        }
        
        .stButton button:hover {
            background-color: #B49B7F; /* Darker taupe on hover */
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #6D5B4B; /* Dark taupe for headers */
        }
        
        /* Code blocks with syntax highlighting */
        code {
            background-color: #F2EBE3; /* Light beige for code */
            border: 1px solid #E0D5C5; /* Subtle border */
        }
        
        pre {
            background-color: #F2EBE3; /* Light beige for code blocks */
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #E0D5C5;
        }
    </style>
    """

# HTML components
def get_about_html():
    return """
    <div style="background-color: #F7F3EB; padding: 15px; border-radius: 8px; border: 1px solid #D2B48C;">
        <h3 style="color: #6D5B4B;">About</h3>
        <p>This RAG Chatbot uses:</p>
        <ul>
            <li>LangChain for document processing</li>
            <li>ChromaDB for vector storage</li>
            <li>OpenAI API for question answering</li>
        </ul>
    </div>
    """

def get_chat_header_html():
    return """
    <div style="background-color: #F7F3EB; padding: 20px; border-radius: 10px; border: 1px solid #D2B48C;">
        <h2 style="color: #6D5B4B;">Chat with your Documents</h2>
    </div>
    """

def get_config_header_html():
    return '<h3 style="color: #6D5B4B;">Configuration</h3>'

def get_upload_header_html():
    return '<h3 style="color: #6D5B4B;">Upload Documents</h3>'

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Apply CSS styling
st.markdown(get_css(), unsafe_allow_html=True)

# Function to initialize the chatbot
def initialize_chatbot():
    api_key = st.session_state.openai_api_key.strip()
    if not api_key:
        st.error("Please enter your OpenAI API key.")
        return False
    
    try:
        # Initialize the chatbot with the provided API key
        st.session_state.chatbot = RAGChatbot(openai_api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return False

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    if not st.session_state.chatbot:
        st.error("Please initialize the chatbot first.")
        return
    
    with st.spinner("Processing documents..."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Load documents into the chatbot
        try:
            num_chunks = st.session_state.chatbot.load_documents(file_paths=file_paths)
            st.session_state.documents_loaded = True
            st.success(f"Successfully processed {len(uploaded_files)} documents with {num_chunks} chunks.")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Main app layout
st.title("📚 RAG Chatbot for Document Q&A")
st.markdown("""
This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents.
Upload PDF, Word, or Excel files, then ask questions about their content.
""")

# Custom container for chat area
chat_container = st.container()
chat_container.markdown(get_chat_header_html(), unsafe_allow_html=True)

# Sidebar for API key and file upload
with st.sidebar:
    # Configuration header
    st.markdown(get_config_header_html(), unsafe_allow_html=True)
    
    # OpenAI API key input
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        key="openai_api_key",
        help="Enter your OpenAI API key to use the chatbot."
    )
    
    # Initialize button
    init_button = st.button("Initialize Chatbot")
    if init_button:
        if initialize_chatbot():
            st.success("Chatbot initialized successfully!")
    
    st.divider()
    
    # File uploader
    st.markdown(get_upload_header_html(), unsafe_allow_html=True)
        
    uploaded_files = st.file_uploader(
        "Upload PDF, Word, or Excel files",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    # Process button
    if uploaded_files:
        process_button = st.button("Process Documents")
        if process_button:
            process_uploaded_files(uploaded_files)
    
    st.divider()
    
    # Clear documents button
    if st.session_state.documents_loaded:
        if st.button("Clear Documents"):
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_documents()
                st.session_state.documents_loaded = False
                st.success("All documents have been cleared.")
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
    
    st.divider()
    
    # App info
    st.markdown(get_about_html(), unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not st.session_state.chatbot:
            response = "Please initialize the chatbot with your OpenAI API key first."
        elif not st.session_state.documents_loaded:
            response = "Please upload and process documents before asking questions."
        else:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.ask(prompt)
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
        
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Instructions if no documents loaded
if not st.session_state.documents_loaded:
    st.info("👈 Please initialize the chatbot and upload documents to get started.")

# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by Streamlit's execution model
    pass