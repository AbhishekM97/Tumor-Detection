# src/data_loader.py

import os
import nibabel as nib
import numpy as np

class BraTSDataset:
    """ Loading the training data into its own object. 
        \\ TODO Considering normalizing and visualizing the data side by side before visualization. 

    """
    def __init__(self, root_dir):
        """data set class constructor

        Args:
            root_dir (String): Realtive path to data files. 
        """
        self.root_dir = root_dir
        self.patient_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        """
        Returns:
            Integer: Number of patient folders in the training data set.
        """
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """ Loads the selected patient volume and visualizes the different types of MRI scans (Modalities).
            T1 provides good anatomical detail, but less tumor visibility. 
            T1CE: Tumor lights up brightly due to contrast agent, best for detecting enhancing tumor. 
            T2: Fluid (edema, CSF) is bright, helps spot swelling.
            FLAIR: Like T2 but with CSF suppressed, better for detecting tumor boundaries near ventricles. 
            Mask: Is the detected/labeled tumor region. Also known as segmentation.

        Args:
            idx (Integer): index number of the patient folder that is being accessed. 

        Returns:
            _type_: images of slices and the mask. 
        """
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

    def normalize(self, image):
        norm_image = np.zeros_like(image)
        for i in range(image.shape[0]): # per modality
            modality = image[i]
            mean = modality.mean()
            std = modality.std()
            norm_image[i] = (modality - mean) / (std + 1e-8)
        return norm_image
    


if __name__ == "__main__":
    """ Test script for loading and visualizing the images.
    """
    from data_loader import BraTSDataset
    from data_visualizer import VolumeVisualizer

    dataset_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    dataset = BraTSDataset(dataset_path)
    print("Loaded dataset with", len(dataset), "volumes.")

    original_image, mask = dataset.__getitem__(70)
    print("Image shape:", original_image.shape)
    print("Mask shape:", mask.shape)

    # Print statistics of the original image
    print("\nOriginal Image Statistics:")
    for i in range(original_image.shape[0]):
        print(f"Modality {i} - Mean: {original_image[i].mean()}, Std: {original_image[i].std()}")

    # Normalize the image
    normalized_image = dataset.normalize(original_image)
    # Print statistics of the normalized image
    print("\nNormalized Image Statistics:")
    for i in range(normalized_image.shape[0]):
        print(f"Modality {i} - Mean: {normalized_image[i].mean()}, Std: {normalized_image[i].std()}")

    # VolumeVisualizer.show_slices(original_image, mask, slice_indices=[60, 70, 80])
    # VolumeVisualizer.show_slices_side_by_side(original_image,normalized_image, mask, slice_idx=(70))
    # Plot histograms of the original and normalized images
    # VolumeVisualizer.show_histograms(original_image, normalized_image=normalized_image, modality_names=['T1', 'T1CE', 'T2', 'FLAIR'])
    
    # Show all modalities in 3D
    VolumeVisualizer.visualize_all_modalities_3d(original_image, modality_names=['T1', 'T1CE', 'T2', 'FLAIR'])

    # Show segmentation mask in 3D
    VolumeVisualizer.visualize_segmentation_3d(mask)
