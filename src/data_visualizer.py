# src/data_visualizer.py

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

"""_summary_
"""
class VolumeVisualizer:
    @staticmethod
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

    def show_slices_side_by_side(original, normalized, mask, slice_idx):
        modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
        vmin, vmax = -3, 3
        for i in range(4):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(original[i, :, :, slice_idx], cmap='gray')
            axes[0].set_title(f"Original {modalities[i]}")

            axes[1].imshow(normalized[i, :, :, slice_idx], cmap='gray')
            axes[1].set_title(f"Normalized {modalities[i]}")

            axes[2].imshow(mask[:, :, slice_idx], cmap='Reds')
            axes[2].set_title("Segmentation Mask")

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def show_histograms(image, normalized_image=None, modality_names=None):
        """
        Plots histograms of pixel intensity values for each modality.
        Optionally compares original and normalized images.
        """
        num_modalities = image.shape[0]
        modality_names = modality_names or [f'Modality {i}' for i in range(num_modalities)]

        fig, axes = plt.subplots(1, num_modalities, figsize=(20, 4))
        for i in range(num_modalities):
            axes[i].hist(image[i].flatten(), bins=100, alpha=0.5, label='Original', range=(0, 300))  # Adjust if needed
            if normalized_image is not None:
                norm_min = normalized_image[i].min()
                norm_max = normalized_image[i].max()
                print('\nnorm_min = ', norm_min, '\nnorm_max = ', norm_max)
                axes[i].hist(normalized_image[i].flatten(), bins=100, alpha=0.5, label='Normalized', range=(-1, 5))
            axes[i].set_title(modality_names[i])
            axes[i].set_yscale('log')
            axes[i].legend()
        plt.tight_layout()
        plt.show()


    @staticmethod
    def visualize_modality_3d(volume, title="MRI Volume"):
        """
        Renders a 3D brain volume using pyvista.

        Args:
            volume (np.ndarray): 3D array (H, W, D) of a single MRI modality.
            title (str): Title of the visualization.
            cmap (str): Colormap for rendering.
            opacity (str or list): Opacity function ('sigmoid', 'linear', etc.).
        """
        volume = volume  # Replace with your actual data

        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1  # For cell data
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid.cell_data["intensity"] = volume.flatten(order="F")

        plotter = pv.Plotter()
        plotter.add_volume(grid)
        plotter.show()

    @staticmethod
    def visualize_all_modalities_3d(image_4d, modality_names=None):
        """
        Renders all 4 MRI modalities in 3D using pyvista.

        Args:
            image_4d (np.ndarray): 4D array of shape (4, H, W, D).
            modality_names (List[str], optional): Names of each modality.
        """
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        for i in range(image_4d.shape[0]):
            VolumeVisualizer.visualize_modality_3d(image_4d[i], title=modality_names[i])

    @staticmethod
    def visualize_segmentation_3d(mask, title="Segmentation Mask"):
        """
        Renders the segmentation mask in 3D.

        Args:
            mask (np.ndarray): 3D array (H, W, D) with labeled tumor regions.
        """
        VolumeVisualizer.visualize_modality_3d(mask, title=title)
