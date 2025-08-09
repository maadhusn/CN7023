# CN7023 - Plant Disease Classification (MSc Coursework)

A simplified ResNet50-based plant disease classification system for MSc coursework using the PlantVillage dataset.

## Overview

This project implements a deep learning solution for plant disease classification using:
- **ResNet50** pretrained model with custom classifier head
- **ImageFolder** data loading with train/val/test splits (70/15/15)
- **Windows-compatible** paths and operations
- **Comprehensive visualizations** for report generation

## Quick Setup

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow numpy
```

### 2. Dataset Setup
Download the PlantVillage dataset and extract to:
```
C:\PlantVillage\
├── Apple___Apple_scab\
├── Apple___Black_rot\
├── Apple___Cedar_apple_rust\
├── Tomato___Late_blight\
└── ... (other disease classes)
```

### 3. Run Training
```bash
python train.py
```

### 4. Generate Visualizations
```bash
python visualize_results.py
```

## Usage

### Training the Model
The `train.py` script will:
- Load data using ImageFolder from `C:/PlantVillage`
- Split into train/val/test (70/15/15) with random_split
- Train ResNet50 for configurable epochs (default: 25)
- Apply augmentations only to training set
- Save model and training history to `results/`

### Generating Report Visualizations
The `visualize_results.py` script creates all plots needed for your report:
- `results/class_distribution.png` - Dataset class distribution
- `results/dataset_samples.png` - 4×4 grid of sample images
- `results/accuracy_curve.png` - Training/validation/test accuracy
- `results/loss_curve.png` - Training/validation loss
- `results/confusion_matrix.png` - Confusion matrix with percentages
- `results/per_class_accuracy.png` - Per-class accuracy bar chart

**Report text summaries** are printed to console for copy-paste into report sections.

## Configuration

Edit the constants in `train.py` to modify training parameters:
- `N_EPOCHS = 25` - Number of training epochs
- `BATCH_SIZE = 32` - Batch size
- `LEARNING_RATE = 0.001` - Learning rate
- `IMAGE_SIZE = 224` - Input image size

## Model Architecture

- **Base Model**: ResNet50 pretrained on ImageNet
- **Classifier Head**: Single linear layer (2048 → num_classes)
- **Optimizer**: Adam with weight decay
- **Augmentations**: Horizontal flip, rotation, random crop (training only)
- **Normalization**: ImageNet mean/std values

## Windows Compatibility

- All paths use forward slashes or `os.path.join()`
- `num_workers=0` in DataLoaders
- No Unix-specific commands
- Results saved to `results/` directory

## Expected Results

- **Training Time**: ~2-4 hours for 25 epochs (depending on dataset size and hardware)
- **Accuracy**: Typically 85-95% on PlantVillage test set
- **Model Size**: ~100MB for ResNet50 weights
- **Generated Files**: 6 PNG plots + model weights + training metrics JSON

## License

MIT License - see LICENSE file for details.
