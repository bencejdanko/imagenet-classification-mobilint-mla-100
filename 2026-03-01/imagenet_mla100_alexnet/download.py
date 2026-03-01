import os
import zipfile
from huggingface_hub import snapshot_download
from config import Config

def download_dataset():
    """
    Downloads zips from Hugging Face and extracts them to match Config structure.
    """
    config = Config()
    repo_id = "bdanko/imagenet__20class_subset"
    
    print(f"Downloading zipped dataset from Hugging Face: {repo_id}")
    
    # 1. Download only the zips and txt files
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["*.zip", "*.txt"],
        local_dir_use_symlinks=False
    )
    
    # 2. Extract zips to their respective destinations
    extraction_map = {
        'imagenet_train20.zip': config.IMAGE_ROOT,
        'imagenet_val20.zip': config.VAL_IMAGE_ROOT
    }

    for zip_name, target_dir in extraction_map.items():
        if os.path.exists(zip_name):
            print(f"Extracting {zip_name} to {target_dir}...")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Optional: remove the zip after extraction to save space in Colab
            # os.remove(zip_name)
        else:
            print(f"Warning: {zip_name} not found after download.")

    print("Dataset preparation complete.")

# Alias for backward compatibility
download_and_extract_dataset = download_dataset

if __name__ == "__main__":
    download_dataset()


