import os
import torch
import torch.nn as nn
import wandb
from augmentation import get_dataloaders, BatchAugmentor
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

    # Load from checkpoint if exists
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'last_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    criterion_cls = nn.CrossEntropyLoss()
    criterion_batch = nn.CrossEntropyLoss(reduction='none')
    batch_augmentor = BatchAugmentor(p=0.5)
    
    # Initialize WandB
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "architecture": "ResNet10",
        }
    )

    # Prepare checkpoint directory
    if config.SAVE_MODEL:
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train, correct_train = 0, 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            images, target_a, target_b, lam = batch_augmentor(images, labels)

            optimizer.zero_grad()

            # Forward Pass
            logits = model(images)

            # Calculate Loss
            loss_1 = criterion_batch(logits, target_a)
            loss_2 = criterion_batch(logits, target_b)
            loss = (loss_1 * lam + loss_2 * (1. - lam)).mean()
            
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Track Train Acc
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            dominant_target = torch.where(lam > 0.5, target_a, target_b)
            correct_train += predicted.eq(dominant_target).sum().item()

        # --- Validation ---
        val_loss, val_acc = validate(model, val_loader, criterion_cls, device)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- WandB Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # --- Checkpointing ---
        if config.SAVE_MODEL:
            # Save latest
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'last_model.pth'))
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
                wandb.log({"best_val_acc": best_val_acc})
                print(f"New best model saved with accuracy: {val_acc:.2f}%")

        print(f"\n>> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% <<")

    print("Training complete.")
    wandb.finish()
    return model, history

if __name__ == "__main__":
    run_training()

