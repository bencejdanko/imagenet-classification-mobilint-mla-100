"""
Intended to be run in Colab notebook
after mounting Google Drive
"""

import shutil
import zipfile

def download_and_extract_dataset():
    print("Starting dataset preparation...")
    
    # --------- copy over metadata ---------
    train_txt_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_train20.txt'
    val_txt_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_val20.txt'

    train_txt = 'imagenet_train20.txt'
    val_txt = 'imagenet_val20.txt'

    print(f"Copying metadata from {train_txt_path} and {val_txt_path}...")
    shutil.copy(train_txt_path, train_txt)
    shutil.copy(val_txt_path, val_txt)

    # --------- copy over zip files ---------
    train_zip_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_train20.zip'
    val_zip_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_val20.zip'

    train_zip = 'imagenet_train20.zip'
    val_zip = 'imagenet_val20.zip'

    print(f"Copying zip files from {train_zip_path} and {val_zip_path}...")
    shutil.copy(train_zip_path, train_zip)
    shutil.copy(val_zip_path, val_zip)

    # --------- unzip files ---------
    print("Extracting training files...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall('train')

    print("Extracting validation files...")
    with zipfile.ZipFile(val_zip, 'r') as zip_ref:
        zip_ref.extractall('val')
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    download_and_extract_dataset()


