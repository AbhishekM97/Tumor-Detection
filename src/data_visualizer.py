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

    @staticmethod
    def visualize_modalities_side_by_side(image_4d, mask=None, modality_names=None):
        """
        Visualizes each MRI modality side by side: one column for the original modality
        and another column for the modality with the segmentation mask overlay.

        Args:
            image_4d (np.ndarray): 4D array of shape (4, H, W, D).
            mask (np.ndarray, optional): 3D array (H, W, D) of the segmentation mask.
            modality_names (List[str], optional): Names of each modality.
        """
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]

        # Create a figure with subplots
        fig, axes = plt.subplots(num_modalities, 2, figsize=(10, 5 * num_modalities))

        for i in range(num_modalities):
            # Plot the original modality
            axes[i, 0].imshow(image_4d[i, :, :, image_4d.shape[3] // 2], cmap='gray')  # Middle slice
            axes[i, 0].set_title(f"{modality_names[i]} (Original)")
            axes[i, 0].axis('off')

            # Plot the modality with the mask overlay
            if mask is not None:
                overlay = np.where(mask > 0, 1, 0)  # Create a binary mask for overlay
                axes[i, 1].imshow(image_4d[i, :, :, image_4d.shape[3] // 2], cmap='gray')  # Middle slice
                axes[i, 1].imshow(overlay[:, :, image_4d.shape[3] // 2], cmap='Reds', alpha=0.5)  # Overlay mask
                axes[i, 1].set_title(f"{modality_names[i]} (With Mask)")
                axes[i, 1].axis('off')

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
    def visualize_modality_3d(volume, mask=None, title="MRI Volume"):
        """
        Renders a 3D brain volume using pyvista, with an optional mask overlay.

        Args:
            volume (np.ndarray): 3D array (H, W, D) of a single MRI modality.
            mask (np.ndarray, optional): 3D array (H, W, D) of the segmentation mask.
            title (str): Title of the visualization.
        """
        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1  # For cell data
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid.cell_data["intensity"] = volume.flatten(order="F")

        plotter = pv.Plotter()
        plotter.add_volume(grid)  # Adjust opacity as needed

        if mask is not None:
            mask_grid = pv.ImageData()
            mask_grid.dimensions = np.array(mask.shape) + 1  # For cell data
            mask_grid.spacing = (1, 1, 1)
            mask_grid.origin = (0, 0, 0)
            mask_grid.cell_data["mask"] = mask.flatten(order="F")

            # Add the mask with a specific color and opacity
            plotter.add_volume(mask_grid, cmap="Reds", show_scalar_bar=True)

        plotter.show()

    @staticmethod
    def visualize_all_modalities_3d(volume, mask=None, modality_names=None):
        """
        Renders all 4 MRI modalities in 3D using pyvista.

        Args:
            image_4d (np.ndarray): 4D array of shape (4, H, W, D).
            modality_names (List[str], optional): Names of each modality.
        """
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        for i in range(volume.shape[0]):
            VolumeVisualizer.visualize_modality_3d(volume[i],mask=mask, title=modality_names[i])

    @staticmethod
    def visualize_segmentation_3d(mask, title="Segmentation Mask"):
        """
        Renders the segmentation mask in 3D.

        Args:
            mask (np.ndarray): 3D array (H, W, D) with labeled tumor regions.
        """
        VolumeVisualizer.visualize_modality_3d(mask, title=title)

    @staticmethod
    def visualize_all_modalities_with_mask_3d(image_4d, mask=None, modality_names=None):
        """
        Renders all MRI modalities in 3D using pyvista, with an optional mask overlay for each modality.

        Args:
            image_4d (np.ndarray): 4D array of shape (4, H, W, D).
            mask (np.ndarray, optional): 3D array (H, W, D) of the segmentation mask.
            modality_names (List[str], optional): Names of each modality.
        """
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]

        # Create a plotter object
        plotter = pv.Plotter(shape=(1,2))

        for i in range(num_modalities):
            volume = image_4d[i]
            grid = pv.ImageData()
            grid.dimensions = np.array(volume.shape) + 1  # For cell data
            grid.spacing = (1, 1, 1)
            grid.origin = (0, 0, 0)
            grid.cell_data["intensity"] = volume.flatten(order="F")

            # Add the modality volume to the plot
            plotter.subplot(0,0)
            plotter.add_volume(grid, name=modality_names[i])  # Adjust opacity as needed

            # Overlay the mask if provided
            if mask is not None:
                plotter.subplot(0,1)
                mask_grid = pv.ImageData()
                mask_grid.dimensions = np.array(mask.shape) + 1  # For cell data
                mask_grid.spacing = (1, 1, 1)
                mask_grid.origin = (0, 0, 0)
                mask_grid.cell_data["mask"] = mask.flatten(order="F")

                # Add the mask with a specific color and opacity
                plotter.add_volume(mask_grid, cmap="Reds", show_scalar_bar=False)

        plotter.show()

    @staticmethod
    def visualize_modalities_separate_grids(image_4d, mask=None, modality_names=None):
        """
        Renders all MRI modalities in 3D using pyvista, with separate grids for each modality:
        one for the modality and one for the modality with the mask overlay.

        Args:
            image_4d (np.ndarray): 4D array of shape (4, H, W, D).
            mask (np.ndarray, optional): 3D array (H, W, D) of the segmentation mask.
            modality_names (List[str], optional): Names of each modality.
        """
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]

        for i in range(num_modalities):
            # Visualize the original modality
            VolumeVisualizer.visualize_modality_3d(image_4d[i], title=f"{modality_names[i]} (Original)")

            # Visualize the modality with the mask overlay
            VolumeVisualizer.visualize_modality_3d(image_4d[i], mask=mask, title=f"{modality_names[i]} (With Mask)")


    @staticmethod
    def visualize_image_and_mask_3d(image_volume, mask_volume, image_title="MRI Volume", mask_title="MRI + Segmentation"):
        """
        Show a 3D MRI volume side by side with the same volume overlaid with a segmentation mask.

        Args:
            image_volume (np.ndarray): 3D MRI volume (H, W, D).
            mask_volume (np.ndarray): 3D segmentation mask (H, W, D), same shape as image_volume.
            image_title (str): Title for the original image pane.
            mask_title (str): Title for the masked image pane.
        """
        # Normalize image volume for visualization
        image_volume = image_volume.astype(np.float32)
        image_volume = (image_volume - image_volume.min()) / (image_volume.max() - image_volume.min())
        image_volume = image_volume * 255

        # Setup grid
        dims = np.array(image_volume.shape) + 1

        # Create PyVista image data (left: original image)
        image_grid = pv.ImageData()
        image_grid.dimensions = dims
        image_grid.spacing = (1, 1, 1)
        image_grid.origin = (0, 0, 0)
        image_grid.cell_data["intensity"] = image_volume.flatten(order="F")

        # Create image data with overlaid mask
        masked_volume = np.copy(image_volume)
        masked_volume[mask_volume > 0] = 255  # Make tumor regions bright for contrast

        mask_grid = pv.ImageData()
        mask_grid.dimensions = dims
        mask_grid.spacing = (1, 1, 1)
        mask_grid.origin = (0, 0, 0)
        mask_grid.cell_data["intensity"] = masked_volume.flatten(order="F")

        # Create side-by-side view
        plotter = pv.Plotter(shape=(1, 2), border=True)

        # Left pane: original image
        plotter.subplot(0, 0)
        plotter.add_volume(image_grid)
        plotter.add_title(image_title)

        # Right pane: image with segmentation overlay
        plotter.subplot(0, 1)
        plotter.add_volume(mask_grid)
        plotter.add_title(mask_title)

        plotter.show()
