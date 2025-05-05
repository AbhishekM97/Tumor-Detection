import matplotlib.pyplot as plt

# src/data_visualizer.py

import matplotlib.pyplot as plt

class VolumeVisualizer:
    @staticmethod

    """_summary_
    """
    def show_slices(image, mask, slice_indices=None):
        """_summary_

        Args:
            image (image - nib): nib loads image objects that represents neuroimaging data. 

            mask (image - nib): nib loaded image that represent the segment of the brain tumor tissue.
            
            slice_indices (Optional Integer range): If user doesn't specify then,
            all the volumes are considered. Defaults to None.
        """
        if slice_indices is None:
            slice_indices = list(range(image.shape[-1]))  # show all slices
        
        # Loop through the slices.
        for slice_idx in slice_indices:
            # New plot with 5 subplots per slice. 4 modalities and 1 mask. 
            # Flag size ensures a wide layout for readability. 
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            modalities = ['T1', 'T1CE', 'T2', 'FLAIR']

            # loop through and plot each modalitiy for the slice. 
            for i in range(4):
                axes[i].imshow(image[i, :, :, slice_idx], cmap='gray')
                axes[i].set_title(modalities[i])
                axes[i].axis('off')
            # plotting the mask for each patient volume. 
            # The mask is visualized in red. 
            axes[4].imshow(mask[:, :, slice_idx], cmap='Reds')
            axes[4].set_title("Segmentation")
            axes[4].axis('off')

            # Applying spacing for a cleaner layout.
            plt.tight_layout()
            plt.show()