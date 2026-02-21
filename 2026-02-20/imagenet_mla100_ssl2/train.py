import torch
import torch.nn as nn
import torch.optim as optim
from augmentation import train_loader, val_loader
from model import NPUModel
from config import Config
config = Config()
from init_hyperparameters import model, optimizer, device
from visualize import visualize_reconstruction

# Validation
# Tracks Accuracy and Top-5 Accuracy
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_acc = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    return avg_loss, avg_acc

criterion_cls = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()

# This hyperparameter controls how much the model cares about reconstructing the image
# vs predicting the class. 0.5 is a great starting point for MAE architectures.
RECON_WEIGHT = 0.5

for epoch in range(config.NUM_EPOCHS):
    model.train()
    total_train, correct_train = 0, 0

    # 1. Fix the Unpacking (Now expects 3 items from the masked dataloader)
    for i, (masked_images, target_images, labels) in enumerate(train_loader):
        masked_images = masked_images.to(device)
        target_images = target_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 2. Forward Pass: Tell the model we need the reconstruction for training
        logits, reconstructed_img, attention_map = model(masked_images, return_reconstruction=True)

        # 3. Calculate the Dual Loss
        loss_cls = criterion_cls(logits, labels)
        loss_recon = criterion_recon(reconstructed_img, target_images)

        # Combine the losses so the gradients mix in the shared encoder
        loss = loss_cls + (RECON_WEIGHT * loss_recon)

        loss.backward()
        optimizer.step()

        # Track Train Acc (using the logits)
        _, predicted = logits.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] Step [{i+1}/{len(train_loader)}] | "
                  f"Acc: {100.*correct_train/total_train:.2f}% | "
                  f"Cls Loss: {loss_cls.item():.4f} | Recon Loss: {loss_recon.item():.4f}")

    # --- Validation & Mismatch Analysis ---
    # Note: Assuming your val_loader still uses the ORIGINAL ImageNet20Dataset without masking.
    # Validation should always be done on clean, unmasked images.
    model.eval()
    val_correct, val_total = 0, 0
    mismatches = []

    visualize_reconstruction(masked_images, target_images, reconstructed_img, attention_map, epoch+1)

    # --- Validation & Mismatch Analysis ---
    model.eval()
    val_correct, val_total = 0, 0
    mismatches = []

    with torch.no_grad():
        # UNPACK 3 ITEMS: We ignore the masked_images for validation
        for masked_images, target_images, labels in val_loader:
            # We evaluate on the clean target_images
            target_images = target_images.to(device)
            labels = labels.to(device)

            # Forward Pass: return_reconstruction=False because we only care about accuracy
            logits = model(target_images, return_reconstruction=False)
            _, predicted = logits.max(1)

            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            # Collect mismatches
            mask = predicted.cpu() != labels.cpu()
            if mask.any() and len(mismatches) < 3:
                idx = torch.where(mask)[0][0]
                mismatches.append({
                    'img': target_images[idx].cpu(),
                    'true': labels[idx].item(),
                    'pred': predicted[idx].item()
                })

    print(f"\n>> Epoch {epoch+1} Summary: Val Acc: {100.*val_correct/val_total:.2f}% <<\n")
