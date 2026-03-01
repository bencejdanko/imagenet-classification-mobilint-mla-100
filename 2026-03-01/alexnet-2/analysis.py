import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(history):
    """
    Plots the loss and accuracy trends from the training history.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_predictions(model, loader, device):
    """
    Helper to get all predictions and true labels from a loader.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_classification_heatmap(model, val_loader, device, class_names=None):
    """
    Generates and plots a confusion matrix heatmap for classification rates.
    """
    y_true, y_pred = get_predictions(model, val_loader, device)
    
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true labels) to get rates
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Classification Rates Heatmap (Confusion Matrix)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def display_classification_report(model, val_loader, device, class_names=None):
    """
    Prints a detailed classification report.
    """
    y_true, y_pred = get_predictions(model, val_loader, device)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)

def display_model_summary(model, input_size=(1, 3, 240, 240)):
    """
    Displays the model architecture summary.
    Attempts to use 'torchinfo' for a detailed summary, falls back to standard print.
    """
    try:
        from torchinfo import summary
        print(summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "kernel_size"]))
    except ImportError:
        print("\ntorchinfo not installed. For a prettier table, run: !pip install torchinfo")
        print("\nStandard PyTorch Print:")
        print(model)