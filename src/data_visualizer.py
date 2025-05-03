import matplotlib.pyplot as plt

# src/data_visualizer.py

import matplotlib.pyplot as plt

class VolumeVisualizer:
    @staticmethod
    def show_slices(image, mask, slice_indices=None):
        if slice_indices is None:
            slice_indices = list(range(image.shape[-1]))  # show all slices

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