# # src/data_loader.py

# import os
# import nibabel as nib
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# class BraTSDataset(Dataset):
#     """ Loading the training data into its own object. 
#         \\ TODO Considering normalizing and visualizing the data side by side before visualization. 

#     """
#     def __init__(self, root_dir):
#         """data set class constructor

#         Args:
#             root_dir (String): Realtive path to data files. 
#         """
#         self.root_dir = root_dir
#         self.patient_dirs = sorted([
#             os.path.join(root_dir, d) for d in os.listdir(root_dir)
#             if os.path.isdir(os.path.join(root_dir, d))
#         ])

#     def __len__(self):
#         """
#         Returns:
#             Integer: Number of patient folders in the training data set.
#         """
#         return len(self.patient_dirs)

#     def __getitem__(self, idx):
#         """ Loads the selected patient volume and visualizes the different types of MRI scans (Modalities).
#             T1 provides good anatomical detail, but less tumor visibility. 
#             T1CE: Tumor lights up brightly due to contrast agent, best for detecting enhancing tumor. 
#             T2: Fluid (edema, CSF) is bright, helps spot swelling.
#             FLAIR: Like T2 but with CSF suppressed, better for detecting tumor boundaries near ventricles. 
#             Mask: Is the detected/labeled tumor region. Also known as segmentation.

#         Args:
#             idx (Integer): index number of the patient folder that is being accessed. 

#         Returns:
#             Tuple: images of slices and the mask as tensors. 
#         """
#         patient_path = self.patient_dirs[idx]
#         base_name = os.path.basename(patient_path)
#         modalities = ['t1', 't1ce', 't2', 'flair']
#         image = []

#         for mod in modalities:
#             path = os.path.join(patient_path, f"{base_name}_{mod}.nii")
#             img = nib.load(path).get_fdata()
#             image.append(img)

#         image = np.stack(image, axis=0)  # Shape: (4, H, W, D)

#         # Load segmentation mask
#         mask_path = os.path.join(patient_path, f"{base_name}_seg.nii")
#         mask = nib.load(mask_path).get_fdata().astype(np.uint8)

#         # Convert to torch tensors
#         image = torch.tensor(image, dtype=torch.float32)  # Ensure image is float32
#         mask = torch.tensor(mask, dtype=torch.long)  # Mask should be long for classification

#         return image, mask

#     def normalize(self, image):
#         norm_image = np.zeros_like(image)
#         for i in range(image.shape[0]): # per modality
#             modality = image[i]
#             mean = modality.mean()
#             std = modality.std()
#             norm_image[i] = (modality - mean) / (std + 1e-8)
#         return norm_image
    

# def get_data_loaders(train_dir, val_dir, batch_size=1):
#     train_dataset = BraTSDataset(train_dir)
#     val_dataset = BraTSDataset(val_dir)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader


import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import random

class BRATSTransform:
    """Data augmentation transforms for BraTS dataset"""
    def __init__(self, p_flip=0.5, p_rotate=0.5, p_noise=0.3):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
    
    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']
        
        # Random horizontal flip
        if random.random() < self.p_flip:
            image = torch.flip(image, dims=[2])  # Flip height dimension
            seg = torch.flip(seg, dims=[1])
        
        # Random vertical flip
        if random.random() < self.p_flip:
            image = torch.flip(image, dims=[3])  # Flip width dimension
            seg = torch.flip(seg, dims=[2])
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            image = torch.rot90(image, k=k, dims=[2, 3])  # Rotate height and width
            seg = torch.rot90(seg, k=k, dims=[1, 2])
        
        # Add random noise to image
        if random.random() < self.p_noise:
            noise = torch.randn_like(image) * 0.01
            image = image + noise
        
        return {'image': image, 'seg': seg, 'id': sample['id']}


class BRATSDataset(Dataset):
    def __init__(self, root_dir, modalities=('flair', 't1', 't1ce', 't2'), transform=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.subject_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_path)

        image = []
        for mod in self.modalities:
            mod_filename = f"{subject_id}_{mod}.nii"
            mod_path = os.path.join(subject_path, mod_filename)
            mod_img = nib.load(mod_path).get_fdata()
            mod_img = self._normalize(mod_img)
            image.append(mod_img)

        image = np.stack(image, axis=0)

        seg_filename = f"{subject_id}_seg.nii"
        seg_path = os.path.join(subject_path, seg_filename)
        seg = nib.load(seg_path).get_fdata()
        seg[seg == 4] = 3

        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.long)

        sample = {'image': image, 'seg': seg, 'id': subject_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _normalize(self, volume):
        volume = volume.astype(np.float32)
        nonzero = volume[np.nonzero(volume)]
        if len(nonzero) > 0:
            mean = nonzero.mean()
            std = nonzero.std()
            volume = (volume - mean) / std
        return volume


# Snippet of get_data_loaders definition from your data_loader.py

def get_data_loaders(train_dir, batch_size=1, val_split=0.2, train_test_split=False, sample_size=None, test_mode=False, train_transform=None, val_transform=None):
    full_dataset = BRATSDataset(train_dir)

    if sample_size is not None and sample_size < len(full_dataset):
        sampled_indices = torch.randperm(len(full_dataset))[:sample_size]
        sampled_dataset = Subset(full_dataset, sampled_indices.tolist())
    else:
        sampled_dataset = full_dataset

    # Train/test/val splitting logic
    if train_test_split:
        train_size = int(len(sampled_dataset) * (1 - val_split - 0.1))  # Example: 70%
        val_size = int(len(sampled_dataset) * val_split)                # Example: 20%
        test_size = len(sampled_dataset) - train_size - val_size        # Remaining 10%

        train_set, val_set, test_set = torch.utils.data.random_split(
            sampled_dataset, [train_size, val_size, test_size]
        )
        
        # Apply transforms to datasets
        if train_transform:
            train_set.dataset.transform = train_transform
        if val_transform:
            val_set.dataset.transform = val_transform

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print(f'val size: {val_size} train size: {train_size} test size: {test_size}')
        return train_loader, val_loader, test_loader

    # If no test split is requested
    train_size = int(len(full_dataset) * (1 - val_split))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

