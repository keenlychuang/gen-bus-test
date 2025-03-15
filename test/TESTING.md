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

* pull the repository test data with the following

    ```bash
    git lfs pull
    ```
