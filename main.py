from src import data_loader, data_visualizer

if __name__ == "__main__":
    """ Test script for loading and visualizing the images.
    """

    dataset_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    dataset = data_loader.BraTSDataset(dataset_path)
    print("Loaded dataset with", len(dataset), "volumes.")

    original_image, mask = dataset.__getitem__(1)
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

    # data_visualizer.VolumeVisualizer.show_slices(original_image, mask, slice_indices=[60, 70, 80])
    # VolumeVisualizer.show_slices_side_by_side(original_image,normalized_image, mask, slice_idx=(50))
    # Plot histograms of the original and normalized images
    # VolumeVisualizer.show_histograms(original_image, normalized_image=normalized_image, modality_names=['T1', 'T1CE', 'T2', 'FLAIR'])
    
    # Show all modalities in 3D
    #VolumeVisualizer.visualize_all_modalities_3d(original_image, mask=mask, modality_names=['T1', 'T1CE', 'T2', 'FLAIR'])
    for i in range(len(original_image)):
        data_visualizer.VolumeVisualizer.visualize_image_and_mask_3d(original_image[i], mask, image_title="MRI Volume", mask_title="MRI + Segmentation")
    # Show segmentation mask in 3D
    # VolumeVisualizer.visualize_segmentation_3d(mask)

    #VolumeVisualizer.visualize_modalities_separate_grids(original_image, mask, modality_names=['T1', 'T1CE', 'T2', 'FLAIR'])
    