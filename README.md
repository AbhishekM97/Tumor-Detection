# Tumor-Detection
Tumor-Detection: A deep learning-based system for detecting and classifying tumors in medical images using Python and CNNs.

| Component       | Purpose |
|---------------- |---------|
| `encoder1/2`    | Learn features from input and reduce spatial size |
| `pool` layers   | Downsample (increase receptive field) |
| `bottleneck`    | Learn deep features |
| `up` layers     | Upsample to higher resolution |
| `decoder1/2`    | Reconstruct spatial info using skip connections |
| `conv`          | Final output prediction |
| `sigmoid`       | Convert to probability mask for segmentation |

