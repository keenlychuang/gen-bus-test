<div align="center">
  <img src="images/dbuse-banner.svg" width="80%" alt="DBUSE RAG Chatbot">
</div>

# Document Base Unified Search and Extraction (DBUSE)

A lightweight Retrieval-Augmented Generation (RAG) chatbot for processing and querying Word documents, Excel files, and PDFs using Python 3.9, OpenAI API, LangChain, and Chroma. 

* Pronunciation: “DEH-byew-see”
* Alternative: "The Bus" 

## Overview

This project implements a RAG chatbot system that allows users to upload documents (Word, Excel, PDF), process them, and ask questions about their content. The chatbot uses OpenAI's language models to generate accurate answers based on the information contained in the uploaded documents.

The system is designed to be lightweight and easy to use, with a clean Streamlit interface for interacting with the chatbot. It leverages the power of LangChain for document processing and Chroma for vector storage, making it efficient and scalable.

## Features

This RAG chatbot system offers the following key features:

- Document processing for multiple file formats:
  - PDF documents (.pdf)
  - Word documents (.docx)
  - Excel spreadsheets (.xlsx, .xls)

- Text extraction and chunking to prepare documents for embedding

- Vector database storage using Chroma for efficient similarity search

- Integration with OpenAI API endpoints for high-quality question answering

- User-friendly Streamlit interface for document uploading and chatting

- Comprehensive demo notebook for exploring the system's capabilities

- Conversational context to allow a more natural style for information retrieval

## Architecture

The system is built with a modular architecture consisting of the following components:

### Document Processing Module

The document processing module is responsible for extracting text from different file formats and splitting it into manageable chunks. It uses LangChain document loaders for each file type:

- PyPDFLoader for PDF documents
- Docx2txtLoader for Word documents
- UnstructuredExcelLoader for Excel spreadsheets

The extracted text is then split into chunks with configurable size and overlap parameters to optimize for context retrieval.

### Vector Database

The vector database module uses Chroma to store and retrieve document embeddings. It leverages OpenAI's embedding models to convert text chunks into vector representations, which are then stored in a persistent Chroma database. This enables efficient similarity search when answering questions.

### RAG Chatbot Core

The core chatbot module integrates the document processor and vector database with OpenAI's language models. It implements a retrieval-augmented generation pipeline that:

1. Retrieves relevant document chunks based on the user's question
2. Constructs a prompt with the retrieved context
3. Generates an answer using OpenAI's language models

### Streamlit UI

The user interface is built with Streamlit, providing an intuitive way to interact with the chatbot. Users can:

- Enter their OpenAI API key
- Upload documents in various formats
- Ask questions about the uploaded documents
- View chat history
- Clear documents or chat history as needed

## Installation

### Prerequisites

- Python 3.9
- Conda (for environment management)

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Create and activate a conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate rag_chatbot
   ```

3. Set up your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

   Alternatively, you can provide your API key directly in the Streamlit interface.

## Usage

### Running the Streamlit App

To launch the Streamlit interface, run:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

This will start the web interface where you can:

1. Enter your OpenAI API key
2. Initialize the chatbot
3. Upload documents (PDF, Word, Excel)
4. Process the documents
5. Ask questions about the document content

### Using the Demo Notebook

The project includes a comprehensive demo notebook that walks through all the functionality:

```bash
jupyter notebook demo_notebook.ipynb
```

The notebook demonstrates:

- Setting up the environment
- Processing sample documents
- Using the vector database
- Asking questions with the RAG chatbot
- Examples of how to use the system with your own documents

### Using the Python API

You can also use the RAG chatbot programmatically in your own Python code:

```python
from rag_chatbot import RAGChatbot

# Initialize the chatbot
chatbot = RAGChatbot(openai_api_key="your-api-key")

# Load documents
chatbot.load_documents(file_paths=["document.pdf", "spreadsheet.xlsx"])

# Ask questions
answer = chatbot.ask("What information is in these documents?")
print(answer)
```

## Project Structure

```bash
rag_chatbot/
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing module
│   └── vector_store.py        # Vector database module
├── app.py                     # Streamlit UI
├── rag_chatbot.py             # Core chatbot implementation
├── run_app.py                 # Helper script to run the Streamlit app
├── demo_notebook.ipynb        # Demo Jupyter notebook
├── test_document_processor.py # Test script for document processor
├── test_vector_store.py       # Test script for vector store
├── test_rag_chatbot.py        # Test script for RAG chatbot
└── README.md                  # This documentation
```

## How It Works

The RAG chatbot operates through the following process:

1. **Document Processing**: When you upload documents, the system extracts text from them using format-specific libraries. The extracted text is then split into smaller chunks with some overlap to maintain context across chunks.

2. **Embedding and Storage**: Each text chunk is converted into a vector embedding using OpenAI's embedding models. These embeddings capture the semantic meaning of the text and are stored in a Chroma vector database for efficient retrieval.

3. **Question Answering**: When you ask a question, the system:
   - Converts your question into an embedding
   - Searches the vector database for the most similar text chunks
   - Retrieves the relevant chunks and constructs a context
   - Sends the question and context to OpenAI's language model
   - Returns the generated answer based on the provided context

This approach combines the benefits of retrieval-based and generation-based methods, resulting in answers that are both relevant to your documents and expressed in natural language.

## Customization

The RAG chatbot can be customized in several ways:

- **Chunk Size and Overlap**: Adjust these parameters in the DocumentProcessor to optimize for your specific documents.

- **Embedding Model**: Change the embedding model in the VectorStore class to use different OpenAI embedding models.

- **Language Model**: Modify the model_name parameter in the RAGChatbot class to use different OpenAI language models.

- **Prompt Template**: Customize the prompt template in the RAGChatbot class to change how context is presented to the language model.

## Limitations

While the RAG chatbot is powerful, it has some limitations to be aware of:

- It requires an OpenAI API key and incurs usage costs based on the number of tokens processed.

- The quality of answers depends on the quality and relevance of the uploaded documents.

- Very large documents may need to be split into smaller files for optimal processing.

- The system works best with text-heavy documents; heavily visual content may not be fully captured.

## Future Improvements

Potential enhancements for future versions:

- Support for more document formats (e.g., HTML, Markdown, CSV, TXT)
- Support for OCR, captioning for images and images within documents
- Support for ipynb tutorials
- Integration with additional language models beyond OpenAI, like Claude
   - Claude DBUSE has a nice ring to it
- Improved handling of tables and structured data from Excel files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project leverages several powerful open-source libraries:

- [LangChain](https://github.com/langchain-ai/langchain) for document processing and RAG pipeline
- [Chroma](https://github.com/chroma-core/chroma) for vector database functionality
- [Streamlit](https://github.com/streamlit/streamlit) for the user interface
- [OpenAI](https://github.com/openai/openai-python) for language models and embeddings

## Contact

For questions or feedback, please open an issue on the GitHub repository or contact the project maintainer.
