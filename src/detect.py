import os
import torch
from torch.utils.data import DataLoader
from data_loader import BraTSDataset
from unet3d import UNet3D
import torch.nn as nn

def evaluate():
    # Settings
    val_dir = "data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # Adjust based on your dataset

    # Load the validation dataset
    val_dataset = BraTSDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize the model
    model = UNet3D(in_channels=4, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/unet3d_epoch_50.pth"))  # Load the latest model

    model.eval()  # Set the model to evaluation mode
    total_accuracy = 0.0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate accuracy (you can implement a more detailed accuracy calculation)
            preds = torch.argmax(outputs, dim=1)
            total_accuracy += (preds == masks).float().sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / (len(val_loader) * masks.numel())  # Total number of pixels

    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
