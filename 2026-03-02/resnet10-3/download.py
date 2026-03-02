import os
import zipfile
from huggingface_hub import snapshot_download
from config import Config

def download_dataset():
    """
    Downloads zips from Hugging Face and extracts them to match Config structure.
    """
    config = Config()
    repo_id = "bdanko/imagenetsubset20"
    
    print(f"Downloading zipped dataset from Hugging Face: {repo_id}")
    
    # Create base directory if it doesn't exist
    if not os.path.exists(config.BASE_DIR):
        print(f"Creating base directory: {config.BASE_DIR}")
        os.makedirs(config.BASE_DIR, exist_ok=True)

    # 1. Download zips and txt files to the absolute base directory
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=config.BASE_DIR,
        allow_patterns=["*.zip", "*.txt"],
        local_dir_use_symlinks=False
    )
    
    # 2. Extract zips using absolute paths
    extraction_map = {
        'imagenet_train20.zip': config.BASE_DIR,
        'imagenet_val20.zip': config.BASE_DIR
    }

    for zip_name, target_dir in extraction_map.items():
        zip_path = os.path.join(config.BASE_DIR, zip_name)
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {target_dir}...")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        else:
            print(f"Warning: {zip_path} not found after download.")

    print("Dataset preparation complete.")

# Alias for backward compatibility
download_and_extract_dataset = download_dataset

if __name__ == "__main__":
    download_dataset()


