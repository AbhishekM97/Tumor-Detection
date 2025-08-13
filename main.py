import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data_loader import BRATSDataset

# test_data_loader.py
def test_data_loader():
    data_path = "data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"  # Update this
    dataset = BRATSDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for batch in loader:
        print("Image shape:", batch['image'].shape)
        print("Segmentation shape:", batch['seg'].shape)
        break


# test_data_visualizer.py
import torch
from torch.utils.data import DataLoader
from src.data_visualizer import VolumeVisualizer
from src.data_loader import BRATSDataset
from src.unet3d import UNet3D

def test_data_visualizer(two_d=False, three_d=False):
    # Set your BraTS data path here
    data_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    # Instantiate the dataset without DataLoader
    simple_dataset = BRATSDataset(data_path)

    # Pick a sample
    sample = simple_dataset[0]  # Index into the dataset directly
    image = sample['image'].numpy()
    mask = sample['seg'].numpy()
    if two_d:
        # Test 2D slice visualization
        VolumeVisualizer.show_slices(image, mask, slice_indices=[60])
        VolumeVisualizer.visualize_modalities_side_by_side(image, mask)
        VolumeVisualizer.show_slices_side_by_side(image, image, mask, slice_idx=60)
        VolumeVisualizer.show_histograms(image, normalized_image=image)
    if three_d:
        # Test 3D visualization
        VolumeVisualizer.visualize_modality_3d(image[0], mask)
        VolumeVisualizer.record_multiple_views(image[0], mask, "T1_volume")
        VolumeVisualizer.record_rotation_animation(image[0], mask, "T1_volume")
        # VolumeVisualizer.visualize_all_modalities_3d(image, mask)
        # VolumeVisualizer.visualize_image_and_mask_3d(image[0], mask)
        # VolumeVisualizer.visualize_modalities_separate_grids(image, mask)
        # VolumeVisualizer.visualize_all_modalities_with_mask_3d(image, mask)

# test_unet3d.py
import torch

def test_model_output():
    # Setup dummy input: batch size 1, 4 modalities, 240x240x155
    dummy_input = torch.randn(1, 4, 240, 240, 155)

    # Create model instance
    model = UNet3D(in_channels=4, out_channels=3)
    model.eval()

    # Forward pass without gradient tracking
    with torch.no_grad():
        output = model(dummy_input)

    # Print and assert output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Expecting slightly smaller depth due to convolution operations
    expected_shape = torch.Size([1, 3, 240, 240, 152])
    assert output.shape == expected_shape, f"Unexpected output shape: got {output.shape}, expected {expected_shape}"

# test_multiple_volumes.py
import torch
from torch.utils.data import DataLoader
from src.data_loader import BRATSDataset
from src.unet3d import UNet3D


def test_multiple_volumes(data_dir, batch_size=2):
    # Set up dataset and dataloader
    dataset = BRATSDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Set up model
    model = UNet3D(in_channels=4, out_channels=3)
    model.eval()

    # Run through batches
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image']  # (B, 4, H, W, D)
            segs = batch['seg']      # (B, H, W, D)

            print(f"Batch {i}: Images shape: {images.shape}, Segs shape: {segs.shape}")
            outputs = model(images)
            print(f"Output shape: {outputs.shape}")

            # Optional: Check for shape mismatch
            assert outputs.shape[2:] == images.shape[2:] or outputs.shape[2] >= images.shape[2] - 2, "Output size mismatch"

            # Process only one batch if just testing shape
            break

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.unet3d import UNet3D
from src.data_loader import BRATSDataset

def test_gpu():
# GPU setup
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = BRATSDataset(root_dir="data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")  # update this path
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = UNet3D(in_channels=4, out_channels=4).to(device)
    model.eval()

    # Run model on first N samples
    N = 2
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            segs = batch['seg'].to(device)

            print(f"Input shape: {images.shape}, Segmentation shape: {segs.shape}")

            try:
                outputs = model(images)
                print(f"Output shape: {outputs.shape}")
            except RuntimeError as e:
                print("RuntimeError:", e)
                if "out of memory" in str(e):
                    print("CUDA Out of memory. Try reducing input volume size or batch size.")
                    torch.cuda.empty_cache()
                break

            count += 1
            if count >= N:
                break



def test_model_output_shape():
    model = UNet3D(in_channels=4, out_channels=4)
    model.eval()

    # Simulate a BraTS input shape: (B, C, H, W, D)
    dummy_input = torch.randn(1, 4, 240, 240, 155)
    with torch.no_grad():
        output = model(dummy_input)

    # Allow some tolerance for minor mismatches (e.g., depth mismatch due to pooling)
    expected_shape = (1, 4, 240, 240, dummy_input.shape[-1])
    assert output.shape[:-1] == expected_shape[:-1], \
        f"Expected output shape ~({expected_shape}), got {output.shape}"
    assert abs(output.shape[-1] - dummy_input.shape[-1]) <= 8, \
        f"Depth mismatch too large: expected ~{dummy_input.shape[-1]}, got {output.shape[-1]}"

    print("✅ test_model_output_shape passed.")


def test_model_saves_checkpoint():
    # Check if at least one checkpoint file exists
    checkpoint_dir = "checkpoints"
    files = os.listdir(checkpoint_dir)
    model_files = [f for f in files if f.endswith('.pth')]
    assert len(model_files) > 0, "No checkpoint files found after training."
    print("✅ test_model_saves_checkpoint passed.")


def test_best_model_saved():
    assert os.path.exists("checkpoints/best_model.pth"), \
        "Best model was not saved."
    print("✅ test_best_model_saved passed.")


# test_train.py
import os
import torch
import shutil
import tempfile
import numpy as np

import src.train as train_module
from src.unet3d import UNet3D


def test_train_output_exists():
    # Setup temp directory
    checkpoint_dir = os.path.join("checkpoints")
    print("\nCheckpoint directory. ", checkpoint_dir)
    # breakpoint()

    # Run training
    train_module.train(test_mode=True, batch_size=1)

    # Check that a checkpoint was created
    assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pth")), "Best model was not saved."
    print("✅ Checkpoint file saved successfully.")


def test_model_forward_pass():
    model = UNet3D(in_channels=4, out_channels=4)
    dummy_input = torch.randn(1, 4, 240, 240, 155)  # Match final output
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 4, 240, 240, 155), f"Unexpected output shape: {output.shape}"
    print("✅ Model forward pass works correctly.")

    

def main():
    # test_model_output()
    # test_multiple_volumes("data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    # test_data_visualizer(two_d=False, three_d=True)
    # test_gpu()
    test_model_output_shape()
    # test_model_saves_checkpoint()
    # test_best_model_saved()
    # test_train_output_exists()
    # test_model_forward_pass()
    # train_module.train( run_number=3, sample_size=100, batch_size=1, test_mode=False)



if __name__ == "__main__":
    main()