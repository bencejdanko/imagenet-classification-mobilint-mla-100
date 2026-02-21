"""
Intended to be run in Colab notebook
after mounting Google Drive
"""

# --------- copy over metadata ---------

import shutil

train_txt_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_train20.txt'
val_txt_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_val20.txt'

train_txt = 'imagenet_train20.txt'
val_txt = 'imagenet_val20.txt'

# Copy training set metadata
shutil.copy(train_txt_path, train_txt)

# Copy validation set metadata
shutil.copy(val_txt_path, val_txt)

# --------- copy over zip files ---------

train_zip_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_train20.zip'
val_zip_path = '/content/drive/MyDrive/NPULab/spring-2026-data/imagenet_val20.zip'

train_zip = 'imagenet_train20.zip'
val_zip = 'imagenet_val20.zip'

shutil.copy(train_zip_path, train_zip)
shutil.copy(val_zip_path, val_zip)

# --------- unzip files ---------

import zipfile

with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall('train')

with zipfile.ZipFile(val_zip, 'r') as zip_ref:
    zip_ref.extractall('val')


