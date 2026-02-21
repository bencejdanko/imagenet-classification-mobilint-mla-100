import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageNet20Dataset

from config import Config

config = Config()

# alter the training dataset for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.INPUT_SHAPE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
])

# keep validation unaltered
transform_val = transforms.Compose([
    transforms.Resize(config.INPUT_SHAPE),
    transforms.ToTensor(),
])

train_dataset = ImageNet20Dataset(txt_file=config.TRAIN_LIST, root_dir=config.IMAGE_ROOT, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
print(f"Dataset loaded: {len(train_dataset)} images found.")


val_dataset = ImageNet20Dataset(txt_file=config.VAL_LIST, root_dir=config.VAL_IMAGE_ROOT, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
print(f"Dataset loaded: {len(val_dataset)} images found.")
