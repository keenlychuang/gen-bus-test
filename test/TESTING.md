# Testing Guide

## Test Data Management

This repository uses Git LFS (Large File Storage) to manage test datasets.

### Setting Up Git LFS

1. Install Git LFS from [the git lfs website](https://git-lfs.com) or with one of the following:

    ```bash
    # macOS (with Homebrew)
    brew install git-lfs

    # Ubuntu/Debian
    sudo apt install git-lfs

    # Windows (with Chocolatey)
    choco install git-lfs

    # Or download from https://git-lfs.com
    ```

2. Initialize Git LFS in your local repository:

   ```bash
   git lfs install
   ```

### Using Git LFS

* Pull the repository test data with:

    ```bash
    git lfs pull
    ```

* Adding new files to Git LFS:

    ```bash
    # Track file patterns in LFS
    git lfs track "*.pdf" "*.csv" "*.xlsx"
    
    # Make sure .gitattributes is committed
    git add .gitattributes
    
    # Add and commit your large files normally
    git add path/to/large/file.pdf
    git commit -m "Add test dataset"
    git push
    ```

* Updating LFS files:

    ```bash
    # Replace or modify LFS-tracked files as needed
    # Then commit changes normally
    git add path/to/updated/file.csv
    git commit -m "Update test dataset with new values"
    git push
    ```

* Checking LFS status:

    ```bash
    # See which files are tracked by LFS
    git lfs ls-files
    
    # Check status of LFS objects
    git lfs status
    ```

## Running Tests

To test the components of the RAG Chatbot system, run the provided test scripts:

```bash
# Test document processor
python test/test_document_processor.py

# Test vector store
python test/test_vector_store.py

# Test RAG chatbot
python test/test_rag_chatbot.py
```

Note: The vector store and RAG chatbot tests require an OpenAI API key to be set.
