import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.data_loader import get_data_loaders, BRATSDataset
from src.unet3d import UNet3D
from torch.cuda.amp import GradScaler, autocast

def dice_score(pred, target, epsilon=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred == target).float().sum()
    return 2. * intersection / (pred.numel() + target.numel() + epsilon)

def save_model(state, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(state, path)

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    dice_total = 0.0

    with torch.no_grad():
        for batch in loader:
            images, masks = batch['image'].to(device), batch['seg'].to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            dice_total += dice_score(outputs, masks).item()

    avg_loss = total_loss / len(loader)
    avg_dice = dice_total / len(loader)
    return avg_loss, avg_dice


def visualize_volume_prediction(model, dataset, device, idx=0, epoch=0, save_dir="epoch_visuals"):
    model.eval()
    sample = dataset[idx]
    image = sample['image'].unsqueeze(0).to(device)
    true_mask = sample['seg'].cpu().numpy()

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model(image)
        pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    volume_depth = true_mask.shape[-1]
    for slice_idx in range(0, volume_depth, 10):  # Every 10th slice to avoid clutter
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(true_mask[:, :, slice_idx], cmap='nipy_spectral')
        axes[0].set_title(f"Ground Truth - Slice {slice_idx}")
        axes[1].imshow(pred_mask[:, :, slice_idx], cmap='nipy_spectral')
        axes[1].set_title(f"Prediction - Slice {slice_idx}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch}_slice_{slice_idx}.png")
        plt.close()


def train(test_mode=False, batch_size=1):
    train_dir = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    num_epochs = 30 if not test_mode else 2
    lr = 1e-4
    print(f"In train.py\nnum_epochs {num_epochs}\ntest_mode: {test_mode}\n")    
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

    # Train/val/test split
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, batch_size=batch_size, val_split=0.2, train_test_split=True, sample_size=20, test_mode=test_mode)

    model = UNet3D(in_channels=4, out_channels=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_dice = 0.0
    accumulation_steps = 4

    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch}")
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)  # Get the total number of batches

        for i, batch in enumerate(train_loader):
            images, masks = batch['image'].to(device), batch['seg'].to(device)
            images.requires_grad_()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            
            total_loss += loss.item()

        # Normalize the total loss by the number of batches
        average_loss = total_loss / num_batches
        val_loss, val_dice = evaluate(model, val_loader, device, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {average_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_model({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, f"checkpoints/best_model_epoch{epoch+1}.pth")

        
        scheduler.step(val_dice)
        visualize_volume_prediction(model, val_loader.dataset, device, idx=0, epoch=epoch+1)


    save_model(model.state_dict(), "checkpoints/final_model.pth")

    print("\nEvaluating on Test Set:")
    test_loss, test_dice = evaluate(model, test_loader, device, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Dice Score: {test_dice:.4f}")
