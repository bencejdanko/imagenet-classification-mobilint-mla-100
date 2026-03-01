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
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train, correct_train = 0, 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward Pass
            logits = model(images)

            # Calculate Loss
            loss = criterion_cls(logits, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Track Train Acc
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # --- Validation ---
        val_loss, val_acc = validate(model, val_loader, criterion_cls, device)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n>> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% <<\n")

    print("Training complete.")
    return model, history

if __name__ == "__main__":
    run_training()

