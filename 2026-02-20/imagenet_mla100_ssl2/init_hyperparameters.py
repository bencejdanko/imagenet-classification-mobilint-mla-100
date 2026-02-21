
import torch
import torch.nn as nn
import torch.optim as optim
from augmentation import train_loader, val_loader
from model import NPUModel
from config import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = NPUModel().to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
