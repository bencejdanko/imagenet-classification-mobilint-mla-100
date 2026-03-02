import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageNet20Dataset
from config import Config
import random
from augmentations.fmix import apply_fmix
from augmentations.cutmix import apply_cutmix
from augmentations.resize_mix import apply_resizemix
from augmentations.mixup import apply_mixup

class BatchAugmentor:
    """Apply batch-level augmentations by standard choice mode over custom rules."""
    def __init__(self, mode='none', p=0.5):
        self.mode = mode
        self.p = p

    def __call__(self, images, labels):
        # By default, pass through
        if self.mode == 'none' or random.random() > self.p:
            device = images.device
            return images, labels, labels, torch.ones(images.size(0), device=device)

        if self.mode == 'mixup':
            res = apply_mixup(images, labels)
            return res[0], res[1], res[2], res[3]
        elif self.mode == 'cutmix':
            res = apply_cutmix(images, labels)
            return res[0], res[1], res[2], res[3]
        elif self.mode == 'fmix':
            res = apply_fmix(images, labels)
            return res[0], res[1], res[2], res[3]
        elif self.mode == 'resizemix':
            res = apply_resizemix(images, labels)
            return res[0], res[1], res[2], res[3]
        elif self.mode == 'hmix':
            from augmentations.hmix import apply_hmix
            res = apply_hmix(images, labels)
            return res[0], res[1], res[2], res[3]
        else:
            device = images.device
            return images, labels, labels, torch.ones(images.size(0), device=device)

def get_dataloaders():
    config = Config()

    # alter the training dataset for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.INPUT_SHAPE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # keep validation unaltered
    transform_val = transforms.Compose([
        transforms.Resize(config.INPUT_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_dataset = ImageNet20Dataset(txt_file=config.TRAIN_LIST, root_dir=config.IMAGE_ROOT, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"Training dataset loaded: {len(train_dataset)} images found.")

    val_dataset = ImageNet20Dataset(txt_file=config.VAL_LIST, root_dir=config.VAL_IMAGE_ROOT, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Validation dataset loaded: {len(val_dataset)} images found.")

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
