# src/data_visualizer.py

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

class VolumeVisualizer:
    @staticmethod
    def show_slices(image, mask, slice_indices=None):
        if slice_indices is None:
            slice_indices = list(range(image.shape[-1]))
        for slice_idx in slice_indices:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
            for i in range(4):
                axes[i].imshow(image[i, :, :, slice_idx], cmap='gray')
                axes[i].set_title(modalities[i])
                axes[i].axis('off')
            axes[4].imshow(mask[:, :, slice_idx], cmap='Reds')
            axes[4].set_title("Segmentation")
            axes[4].axis('off')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def visualize_modalities_side_by_side(image_4d, mask=None, modality_names=None):
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]
        fig, axes = plt.subplots(num_modalities, 2, figsize=(10, 5 * num_modalities))
        for i in range(num_modalities):
            axes[i, 0].imshow(image_4d[i, :, :, image_4d.shape[3] // 2], cmap='gray')
            axes[i, 0].set_title(f"{modality_names[i]} (Original)")
            axes[i, 0].axis('off')
            if mask is not None:
                overlay = np.where(mask > 0, 1, 0)
                axes[i, 1].imshow(image_4d[i, :, :, image_4d.shape[3] // 2], cmap='gray')
                axes[i, 1].imshow(overlay[:, :, image_4d.shape[3] // 2], cmap='Reds', alpha=0.5)
                axes[i, 1].set_title(f"{modality_names[i]} (With Mask)")
                axes[i, 1].axis('off')
        plt.tight_layout()
        plt.show()

    def show_slices_side_by_side(original, normalized, mask, slice_idx):
        modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
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
        num_modalities = image.shape[0]
        modality_names = modality_names or [f'Modality {i}' for i in range(num_modalities)]
        fig, axes = plt.subplots(1, num_modalities, figsize=(20, 4))
        for i in range(num_modalities):
            axes[i].hist(image[i].flatten(), bins=100, alpha=0.5, label='Original', range=(0, 300))
            if normalized_image is not None:
                axes[i].hist(normalized_image[i].flatten(), bins=100, alpha=0.5, label='Normalized', range=(-1, 5))
            axes[i].set_title(modality_names[i])
            axes[i].set_yscale('log')
            axes[i].legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_modality_3d(volume, mask=None, title="MRI Volume"):
        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid.cell_data["intensity"] = volume.flatten(order="F")
        plotter = pv.Plotter()
        plotter.add_volume(grid)
        if mask is not None:
            mask_grid = pv.ImageData()
            mask_grid.dimensions = np.array(mask.shape) + 1
            mask_grid.spacing = (1, 1, 1)
            mask_grid.origin = (0, 0, 0)
            mask_grid.cell_data["mask"] = mask.flatten(order="F")
            plotter.add_volume(mask_grid, cmap="Reds", show_scalar_bar=True)
        plotter.show()

    @staticmethod
    def visualize_all_modalities_3d(volume, mask=None, modality_names=None):
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        for i in range(volume.shape[0]):
            VolumeVisualizer.visualize_modality_3d(volume[i], mask=mask, title=modality_names[i])

    @staticmethod
    def visualize_segmentation_3d(mask, title="Segmentation Mask"):
        VolumeVisualizer.visualize_modality_3d(mask, title=title)

    @staticmethod
    def visualize_all_modalities_with_mask_3d(image_4d, mask=None, modality_names=None):
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]
        plotter = pv.Plotter(shape=(1, 2))
        for i in range(num_modalities):
            volume = image_4d[i]
            grid = pv.ImageData()
            grid.dimensions = np.array(volume.shape) + 1
            grid.spacing = (1, 1, 1)
            grid.origin = (0, 0, 0)
            grid.cell_data["intensity"] = volume.flatten(order="F")
            plotter.subplot(0, 0)
            plotter.add_volume(grid, name=modality_names[i])
            if mask is not None:
                plotter.subplot(0, 1)
                mask_grid = pv.ImageData()
                mask_grid.dimensions = np.array(mask.shape) + 1
                mask_grid.spacing = (1, 1, 1)
                mask_grid.origin = (0, 0, 0)
                mask_grid.cell_data["mask"] = mask.flatten(order="F")
                plotter.add_volume(mask_grid, cmap="Reds", show_scalar_bar=False)
        plotter.show()

    @staticmethod
    def visualize_modalities_separate_grids(image_4d, mask=None, modality_names=None):
        modality_names = modality_names or ['T1', 'T1CE', 'T2', 'FLAIR']
        num_modalities = image_4d.shape[0]
        for i in range(num_modalities):
            VolumeVisualizer.visualize_modality_3d(image_4d[i], title=f"{modality_names[i]} (Original)")
            VolumeVisualizer.visualize_modality_3d(image_4d[i], mask=mask, title=f"{modality_names[i]} (With Mask)")

    @staticmethod
    def visualize_image_and_mask_3d(image_volume, mask_volume, image_title="MRI Volume", mask_title="MRI + Segmentation"):
        image_volume = image_volume.astype(np.float32)
        image_volume = (image_volume - image_volume.min()) / (image_volume.max() - image_volume.min()) * 255
        dims = np.array(image_volume.shape) + 1
        image_grid = pv.ImageData()
        image_grid.dimensions = dims
        image_grid.spacing = (1, 1, 1)
        image_grid.origin = (0, 0, 0)
        image_grid.cell_data["intensity"] = image_volume.flatten(order="F")
        masked_volume = np.copy(image_volume)
        masked_volume[mask_volume > 0] = 255
        mask_grid = pv.ImageData()
        mask_grid.dimensions = dims
        mask_grid.spacing = (1, 1, 1)
        mask_grid.origin = (0, 0, 0)
        mask_grid.cell_data["intensity"] = masked_volume.flatten(order="F")
        plotter = pv.Plotter(shape=(1, 2), border=True)
        plotter.subplot(0, 0)
        plotter.add_volume(image_grid)
        plotter.add_title(image_title)
        plotter.subplot(0, 1)
        plotter.add_volume(mask_grid)
        plotter.add_title(mask_title)
        plotter.show()
