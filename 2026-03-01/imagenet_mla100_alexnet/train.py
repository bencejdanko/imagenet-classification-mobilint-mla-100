import torch
import torch.nn as nn
from augmentation import get_dataloaders
from init_hyperparameters import initialize_training
from config import Config

# Validation function
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

def run_training(model=None, optimizer=None, device=None, train_loader=None, val_loader=None):
    config = Config()
    
    # Initialize components if not provided
    if model is None or optimizer is None or device is None:
        print("Initializing model, optimizer, and device...")
        model, optimizer, device = initialize_training()
    
    if train_loader is None or val_loader is None:
        print("Loading data...")
        train_loader, val_loader = get_dataloaders()

    criterion_cls = nn.CrossEntropyLoss()

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train, correct_train = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward Pass
            logits = model(images)

            # Calculate Loss
            loss = criterion_cls(logits, labels)

            loss.backward()
            optimizer.step()

            # Track Train Acc
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        mismatches = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                _, predicted = logits.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Collect mismatches for debugging (optional)
                mask = predicted.cpu() != labels.cpu()
                if mask.any() and len(mismatches) < 3:
                    idx = torch.where(mask)[0][0]
                    mismatches.append({
                        'img': images[idx].cpu(),
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item()
                    })

        print(f"\n>> Epoch {epoch+1} Summary: Train Acc: {100.*correct_train/total_train:.2f}% | Val Acc: {100.*val_correct/val_total:.2f}% <<\n")

    print("Training complete.")

if __name__ == "__main__":
    run_training()

