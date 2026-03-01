import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class ImageNet20Dataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.img_labels = []
        self.root_dir = root_dir
        self.transform = transform

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.img_labels.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename, label = self.img_labels[idx]
        path_flat = os.path.join(self.root_dir, filename)
        path_nested = os.path.join(self.root_dir, filename.split('_')[0], filename)
        full_path = path_flat if os.path.exists(path_flat) else path_nested

        if not os.path.exists(full_path):
            print(f"Warning: Image not found at {full_path}. Check your IMAGE_ROOT / VAL_IMAGE_ROOT config.")
            return torch.zeros(3, 224, 224), label

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error opening image {full_path}: {e}")
            image = Image.new('RGB', (240, 240))

        # apply augmentations
        target_image = self.transform(image) if self.transform else image
        
        return target_image, label
