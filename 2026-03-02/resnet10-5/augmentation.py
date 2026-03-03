import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageNet20Dataset
from config import Config
import random

def get_dataloaders():
    config = Config()

    # alter the training dataset for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.INPUT_SHAPE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # keep validation unaltered
    transform_val = transforms.Compose([
        transforms.Resize(config.INPUT_SHAPE),
        transforms.CenterCrop(config.INPUT_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_dataset = ImageNet20Dataset(txt_file=config.TRAIN_LIST, root_dir=config.IMAGE_ROOT, transform=train_transform)

    use_mix = getattr(config, 'USE_MIX_AUGMENTATIONS', False)
    if use_mix:
        from torchvision.transforms import v2
        from torch.utils.data import default_collate
        mix_alpha = getattr(config, 'MIX_ALPHA', 0.2)
        cutmix = v2.CutMix(num_classes=config.NUM_CLASSES, alpha=mix_alpha)
        mixup = v2.MixUp(num_classes=config.NUM_CLASSES, alpha=mix_alpha)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
    print(f"Training dataset loaded: {len(train_dataset)} images found.")

    val_dataset = ImageNet20Dataset(txt_file=config.VAL_LIST, root_dir=config.VAL_IMAGE_ROOT, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Validation dataset loaded: {len(val_dataset)} images found.")

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
