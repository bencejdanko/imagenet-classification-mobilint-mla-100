import os
import torch
import torch.nn as nn
import wandb
from augmentation import get_dataloaders
from init_hyperparameters import initialize_training
from config import Config
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from torch.optim.swa_utils import get_ema_multi_avg_fn

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

    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Initialize Scheduler
    t_0 = getattr(config, 'T_0', 10)
    t_mult = getattr(config, 'T_MULT', 1)
    eta_min = getattr(config, 'ETA_MIN', 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
    )
    
    # Initialize EMA
    use_ema = getattr(config, 'USE_EMA', True)
    ema_decay = getattr(config, 'EMA_DECAY', 0.999)
    if use_ema:
        print(f"Initializing EMA model (decay={ema_decay})...")
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    else:
        ema_model = None
    
    # Initialize WandB
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "architecture": "ResNet10",
            "aug_mode": "trivialaugment",
            "label_smoothing": label_smoothing,
            "T_0": t_0,
            "T_MULT": t_mult,
            "ETA_MIN": eta_min,
            "use_ema": use_ema,
            "ema_decay": ema_decay if use_ema else None,
            "use_mix_augmentations": getattr(config, 'USE_MIX_AUGMENTATIONS', False),
            "mix_alpha": getattr(config, 'MIX_ALPHA', 0.2),
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

            optimizer.zero_grad()

            # Forward Pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Calculate Loss
            loss = criterion_cls(logits, labels)
            
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if use_ema:
                ema_model.update_parameters(model)

            # Track Train Acc
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            
            # Handle soft labels created by mixup/cutmix
            target_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
            correct_train += predicted.eq(target_labels).sum().item()

        # --- Validation ---
        eval_model = ema_model if use_ema else model
        val_loss, val_acc = validate(eval_model, val_loader, criterion_cls, device)
        
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
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        scheduler.step()

        # --- Checkpointing ---
        if config.SAVE_MODEL:
            # We save the eval_model (which is ema_model if use_ema=True, otherwise model)
            save_state = eval_model.module.state_dict() if hasattr(eval_model, 'module') else eval_model.state_dict()
            
            # Save latest
            torch.save(save_state, os.path.join(config.CHECKPOINT_DIR, 'last_model.pth'))
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(save_state, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
                wandb.log({"best_val_acc": best_val_acc})
                print(f"New best model saved with accuracy: {val_acc:.2f}%")

        print(f"\n>> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% <<")

    print("Training complete.")
    wandb.finish()
    return model, history

if __name__ == "__main__":
    run_training()

