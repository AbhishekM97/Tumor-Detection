# 3D Brain Tumor Segmentation with CNNs

## 📌 Overview
This project implements a 3D Convolutional Neural Network for brain tumor segmentation using the **BraTS dataset** from the University of Pennsylvania.  
The model detects and classifies tumor subregions including:
- **NCR** (Necrotic and Non-Enhancing Tumor Core)
- **ED** (Peritumoral Edema)
- **ET** (Enhancing Tumor)

---

## 🎯 Goal
To develop a deep learning model capable of accurately segmenting brain tumors from MRI scans, aiding in early detection and treatment planning.

---

## 🛠 Approach
- **Model**: 3D U-Net-like architecture with ConvBlock modules  
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam + Learning Rate Scheduler (ReduceLROnPlateau)  
- **Training**: Mixed precision, gradient accumulation for memory efficiency  
- **Validation Metric**: Dice Score per class

---

## 📊 Results
| Metric        | Score |
|--------------|-------|
| **Validation Dice** | 0.9947 |
| **Test Dice**       | 0.9271 |
| **NCR Dice**        | 0.9633 |
| **ED Dice**         | 0.0073 |
| **ET Dice**         | 0.4108 |

---

## 🖼 Example Predictions
| Input MRI Slice | Ground Truth | Prediction |
|-----------------|--------------|------------|
| ![slice](images/slice_50.png) | ![gt](images/slice_50_gt.png) | ![pred](images/slice_50_pred.png) |

---

## 🔮 Future Work
- Improve ED segmentation performance
- Incorporate data augmentation
- Experiment with hybrid loss functions (Dice + CE)
- Explore Transformer-based architectures for 3D data

---

## 📂 Repository Structure
├── src/ # Training and evaluation scripts
├── models/ # Model architectures
├── data/ # Data loading utilities (no raw data)
├── outputs/ # Saved model checkpoints & logs
├── images/ # Result visualizations
├── report.pdf # Technical report
├── presentation.pdf # Project presentation slides
├── requirements.txt # Dependencies
└── README.md # Project documentation



---

## 🙏 Acknowledgements
Dataset: [BraTS - University of Pennsylvania](https://www.med.upenn.edu/cbica/brats2020/data.html)
