"""
Intended to be run in Colab notebook
Uses Hugging Face Hub to download the dataset
"""

import os
from huggingface_hub import snapshot_download

def download_dataset():
    """
    Downloads the dataset from Hugging Face Hub.
    Replaces the old manual copy-and-extract logic.
    """
    repo_id = "bdanko/imagenet__20class_subset"
    print(f"Downloading dataset from Hugging Face: {repo_id}")
    
    # This downloads the entire repository to the current directory.
    # It will include the .txt metadata and the image folders/zips.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    
    # If the user still wants to extract zips (if they uploaded zips instead of folders):
    # we can keep a simple check here, but the goal is to have the files ready.
    # For now, we assume the repo structure aligns with Config expectations.
    
    print("Dataset download complete.")

# Alias for backward compatibility in existing notebooks
download_and_extract_dataset = download_dataset

if __name__ == "__main__":
    download_dataset()


