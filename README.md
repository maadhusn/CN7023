# CN7023 - Plant Disease Classification (MSc Coursework)

A comprehensive ResNet50-based plant disease classification system with ANN baseline comparison for MSc coursework using the PlantVillage dataset.

## Overview

This project implements a complete machine learning pipeline for plant disease classification:
- **CNN Model**: ResNet50 pretrained on ImageNet with custom classifier head
- **ANN Baseline**: MLP trained on downsampled (32×32) RGB images for comparison
- **Deterministic Splits**: Manifest-based train/val/test splits (70/15/15) for reproducibility
- **Comprehensive Metrics**: Precision, recall, F1-score, confusion analysis
- **Explainability**: Grad-CAM visualizations for CNN interpretability
- **Windows Compatible**: Forward slash paths, num_workers=0, results/ directory
- **Report Ready**: Automated generation of plots and paste-ready text summaries

## Files Included

### Core Training & Evaluation Scripts
- `train.py` - ResNet50 CNN training with manifest-based data loading
- `eval.py` - Enhanced evaluation with comprehensive metrics and Grad-CAM integration
- `visualize_results.py` - Generate all report-ready plots and text summaries

### ANN Baseline
- `train_ann_baseline.py` - MLP baseline training on downsampled (32×32) images
- `ann_dataset.py` - Dataset loader for downsampled images from manifests

### Explainability & Visualization
- `gradcam.py` - Grad-CAM visualization for CNN interpretability
- `cv_viz.py` - OpenCV prediction overlays with confidence scores

### MATLAB Preprocessing (Optional)
- `matlab/leaf_preprocess.m` - HSV thresholding and morphological operations

### Configuration & Documentation
- `config.yaml` - All training parameters and dataset configuration
- `requirements.txt` - Python dependencies
- `README.md` - Complete documentation and Windows workflow
- `LICENSE` - MIT license

### Output Directories
- `results/` - All generated plots, metrics, and model checkpoints
- `report_assets/` - Additional assets (kept for compatibility)

**Note**: This repository supports only Windows flat layout with dataset at `C:\PlantVillage\`.

## Local Windows Quick Run

```powershell
# from C:\CN7023
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (make sure your dataset is at C:\PlantVillage with class folders inside; no 'splits' needed)
python train.py
python eval.py --gradcam 24
python visualize_results.py

# outputs in .\results\
```

## Quick Setup

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow numpy pyyaml
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

## Run Order (Windows Workflow)

Execute the following commands in order for complete MSc coursework pipeline:

```bat
:: 1) (Optional) ANN baseline (downsampled)
python train_ann_baseline.py --config config.yaml

:: 2) Train CNN (ResNet50)
python train.py --config config.yaml

:: 3) Evaluate + Grad-CAM + metrics
python eval.py --config config.yaml --gradcam 24

:: 4) Export report visuals & text summaries
python visualize_results.py --config config.yaml
```

## Generated Files for Report

After running the complete pipeline, the following files will be available in `results/`:

### Dataset Analysis
- `class_distribution.png` - Bar chart of samples per class
- `dataset_samples.png` - 4×4 grid of sample images from different classes

### Training Results
- `accuracy_curve.png` - Training/validation/test accuracy over epochs
- `loss_curve.png` - Training/validation loss over epochs

### Model Evaluation
- `confusion_matrix.png` - Confusion matrix with per-class accuracy percentages
- `per_class_accuracy.png` - Per-class accuracy bar chart with color coding
- `classification_report.txt` - Detailed sklearn classification report
- `metrics.json` - Comprehensive metrics (precision/recall/F1)
- `top_confusions.csv` - Top 10 confused class pairs by count

### Explainability & Visualization
- `gradcam_correct_*.png` - Grad-CAM visualizations for correctly classified samples
- `gradcam_missed_*__pred_vs_true__basename.png` - Grad-CAM for misclassified samples
- `viz_pred_*.png` - OpenCV prediction overlays with confidence scores

### ANN Baseline (if run)
- `ann_curves.png` - ANN training curves
- `ann_confusion_matrix.png` - ANN confusion matrix
- `per_class_accuracy_ann.png` - ANN per-class accuracy
- `ann_metrics.json` - ANN performance metrics

### Report Text Summaries (Paste-Ready)
- `summary_dataset.txt` - Dataset description for report
- `summary_results.txt` - Results summary with key metrics
- `summary_analysis.txt` - Critical analysis with confusion patterns

## Configuration

Edit parameters in `config.yaml`:

```yaml
seed: 42                    # Reproducibility seed
subset_per_class: null      # Limit samples per class (for development)

dataset:
  path: "C:/PlantVillage"   # Dataset location
  image_size: 224           # Input image size
  train_split: 0.7          # Training set ratio
  val_split: 0.15           # Validation set ratio
  test_split: 0.15          # Test set ratio

training:
  epochs: 25                # CNN training epochs
  batch_size: 32            # Batch size
  learning_rate: 0.001      # Learning rate
  weight_decay: 0.0001      # Weight decay

ann:
  downsample_size: 32       # ANN input image size
  hidden1: 128              # First hidden layer size
  hidden2: 64               # Second hidden layer size
  dropout: 0.2              # Dropout rate
  lr: 0.001                 # ANN learning rate
  epochs: 40                # ANN training epochs

gradcam:
  samples: 24               # Number of Grad-CAM samples
```

## Optional MATLAB Preprocessing

For advanced image preprocessing using MATLAB:

### 1. Run MATLAB Preprocessing
```matlab
% In MATLAB command window
addpath('matlab');
leaf_preprocess('C:\PlantVillage', 'C:\PlantVillage_processed', 'color');
```

### 2. Update Configuration
```yaml
dataset:
  path: "C:/PlantVillage_processed"  # Use processed dataset
```

The MATLAB script applies:
- HSV color space conversion
- S-channel thresholding for vegetation detection
- Morphological operations (opening, closing)
- Largest connected component selection
- Histogram equalization on V channel

## Model Architecture

### CNN (ResNet50)
- **Base Model**: ResNet50 pretrained on ImageNet
- **Classifier Head**: Single linear layer (2048 → num_classes)
- **Optimizer**: Adam with weight decay
- **Augmentations**: Horizontal flip, rotation, random crop (training only)
- **Normalization**: ImageNet mean/std values

### ANN Baseline (MLP)
- **Input**: Flattened 32×32×3 RGB images (3072 features)
- **Architecture**: 3072 → 128 → 64 → num_classes
- **Activation**: ReLU with 0.2 dropout
- **Purpose**: Baseline comparison to demonstrate CNN effectiveness

## Expected Results

- **CNN Accuracy**: Typically 85-95% on PlantVillage test set
- **ANN Accuracy**: Typically 60-75% (demonstrates CNN superiority)
- **Training Time**: ~2-4 hours for CNN, ~30 minutes for ANN
- **Generated Files**: 15+ visualization files + metrics + text summaries

## Windows Compatibility

- All paths use forward slashes or `os.path.join()`
- `num_workers=0` in all DataLoaders
- No Unix-specific commands
- Results saved to `results/` directory
- Batch file commands provided for workflow

## Troubleshooting

- **Memory Issues**: Reduce `batch_size` in config.yaml
- **Slow Training**: Reduce `epochs` or use `subset_per_class`
- **Path Errors**: Ensure dataset is at exact path `C:\PlantVillage\`
- **Missing Files**: Run commands in exact order shown above

## License

MIT License - see LICENSE file for details.
