import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageNet20Dataset
from config import Config
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import random
from augmentations.fmix import apply_fmix
from augmentations.cutmix import apply_cutmix
from augmentations.resize_mix import apply_resizemix
from augmentations.mixup import apply_mixup

class BatchAugmentor:
    """Apply batch-level augmentations like mixup, cutmix, fmix."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, labels):
        # By default, pass through
        if random.random() > self.p:
            device = images.device
            return images, labels, labels, torch.ones(images.size(0), device=device)

        # Determine which mix to apply based on class presence (remediating specific confusions)
        has_15 = (labels == 15).any().item()
        has_animals = ((labels == 9) | (labels == 6) | (labels == 8)).any().item()

        if has_15:
            # Remediate class 15 confusions
            mixed_images, target_a, target_b, actual_lam, mask, rand_index = apply_fmix(images, labels)
            return mixed_images, target_a, target_b, actual_lam
        elif has_animals:
            # Distinguish animals
            if random.random() > 0.5:
                res = apply_cutmix(images, labels)
            else:
                res = apply_resizemix(images, labels)
            return res[0], res[1], res[2], res[3]
        else:
            res = apply_mixup(images, labels)
            return res[0], res[1], res[2], res[3]

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

    # Class aware sampling
    labels = [label for _, label in train_dataset.img_labels]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler)
    print(f"Training dataset loaded: {len(train_dataset)} images found.")

    val_dataset = ImageNet20Dataset(txt_file=config.VAL_LIST, root_dir=config.VAL_IMAGE_ROOT, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Validation dataset loaded: {len(val_dataset)} images found.")

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
