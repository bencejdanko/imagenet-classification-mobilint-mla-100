import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BlockMasking:
    """Drops out random contiguous blocks of the image to force global reasoning."""
    def __init__(self, block_size=16, mask_ratio=0.4):
        self.block_size = block_size
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        # img shape: [3, H, W]
        c, h, w = img.shape
        grid_h, grid_w = h // self.block_size, w // self.block_size

        # Create a randomized binary mask grid
        mask = torch.rand(1, grid_h, grid_w) < self.mask_ratio
        mask = mask.float()

        # Upsample the small grid to the full image size using nearest neighbor
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)

        # 1 means mask it (turn to 0), 0 means keep it.
        masked_img = img * (1 - mask)
        return masked_img

class ImageNet20Dataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.img_labels = []
        self.root_dir = root_dir
        self.transform = transform
        self.masking = BlockMasking(block_size=16, mask_ratio=0.4) # 40% of image dropped

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

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (240, 240))

        # 1. Apply standard augmentations (Resize, ColorJitter, ToTensor)
        target_image = self.transform(image) if self.transform else image

        # 2. Generate the masked version for the input
        masked_image = self.masking(target_image)

        # Return both! Model sees masked_image, loss function uses target_image
        return masked_image, target_image, label
