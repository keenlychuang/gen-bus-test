"""
Styling module for RAG Chatbot Streamlit app
Contains all custom CSS and styling functions
"""

import streamlit as st

def get_light_mode_css():
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

def get_dark_mode_css():
    return """
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #2A2520; /* Dark taupe background */
            color: #F0F0F0; /* Light text */
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-12oz5g7 {
            background-color: #413933; /* Dark tan/taupe sidebar */
        }
        
        /* Chat message containers */
        .stChatMessage {
            background-color: #322A24; /* Dark cream color for messages */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* User messages */
        .stChatMessage[data-testid="user-message"] {
            background-color: #3D3630; /* Dark taupe for user messages */
        }
        
        /* Assistant messages */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #2D2824; /* Darker cream for assistant */
        }
        
        /* Input fields */
        .stTextInput input, .stFileUploader label {
            background-color: #3A332D; /* Dark cream */
            border: 1px solid #5D5047; /* Lighter taupe border */
            color: #F0F0F0;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #4D443C; /* Dark taupe */
            color: #F0F0F0;
            border: none;
        }
        
        .stButton button:hover {
            background-color: #5A5047; /* Lighter taupe on hover */
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #C0B5A8; /* Light taupe for headers */
        }
        
        /* Code blocks with syntax highlighting */
        code {
            background-color: #3D3630; /* Dark beige for code */
            border: 1px solid #4A423B; /* Subtle border */
            color: #E0D5C5;
        }
        
        pre {
            background-color: #3D3630; /* Dark beige for code blocks */
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #4A423B;
            color: #E0D5C5;
        }
        
        /* Info, success, error boxes */
        .stAlert {
            background-color: #3D3630;
            color: #F0F0F0;
        }
    </style>
    """

def get_light_about_html():
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

def get_dark_about_html():
    return """
    <div style="background-color: #3D3630; padding: 15px; border-radius: 8px; border: 1px solid #5D5047;">
        <h3 style="color: #C0B5A8;">About</h3>
        <p>This RAG Chatbot uses:</p>
        <ul>
            <li>LangChain for document processing</li>
            <li>ChromaDB for vector storage</li>
            <li>OpenAI API for question answering</li>
        </ul>
    </div>
    """

def get_light_chat_header_html():
    return """
    <div style="background-color: #F7F3EB; padding: 20px; border-radius: 10px; border: 1px solid #D2B48C;">
        <h2 style="color: #6D5B4B;">Chat with your Documents</h2>
    </div>
    """

def get_dark_chat_header_html():
    return """
    <div style="background-color: #3D3630; padding: 20px; border-radius: 10px; border: 1px solid #5D5047;">
        <h2 style="color: #C0B5A8;">Chat with your Documents</h2>
    </div>
    """

def get_light_config_header_html():
    return '<h3 style="color: #6D5B4B;">Configuration</h3>'

def get_dark_config_header_html():
    return '<h3 style="color: #C0B5A8;">Configuration</h3>'

def get_light_upload_header_html():
    return '<h3 style="color: #6D5B4B;">Upload Documents</h3>'

def get_dark_upload_header_html():
    return '<h3 style="color: #C0B5A8;">Upload Documents</h3>'

def apply_theme(dark_mode=False):
    """Apply the appropriate theme based on dark_mode setting"""
    if dark_mode:
        st.markdown(get_dark_mode_css(), unsafe_allow_html=True)
    else:
        st.markdown(get_light_mode_css(), unsafe_allow_html=True)