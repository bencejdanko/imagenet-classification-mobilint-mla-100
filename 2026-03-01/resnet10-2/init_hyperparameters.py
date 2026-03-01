import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet10
from config import Config

def initialize_training():
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet10(num_classes=config.NUM_CLASSES).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    return model, optimizer, device

if __name__ == "__main__":
    model, optimizer, device = initialize_training()
