# src/data_loader.py

import os
import nibabel as nib
import numpy as np

class BraTSDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]
        base_name = os.path.basename(patient_path)
        modalities = ['t1', 't1ce', 't2', 'flair']
        image = []

        for mod in modalities:
            path = os.path.join(patient_path, f"{base_name}_{mod}.nii")
            img = nib.load(path).get_fdata()
            image.append(img)

        image = np.stack(image, axis=0)  # Shape: (4, H, W, D)

        # Load segmentation mask
        mask_path = os.path.join(patient_path, f"{base_name}_seg.nii")
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        return image, mask




if __name__ == "__main__":
    from data_loader import BraTSDataset
    from data_visualizer import VolumeVisualizer

    dataset_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    dataset = BraTSDataset(dataset_path)

    print("Loaded dataset with", len(dataset), "volumes.")

    image, mask = dataset[0]
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    VolumeVisualizer.show_slices(image, mask, slice_indices=[60, 70, 80])