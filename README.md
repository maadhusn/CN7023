# CN7023 - PlantVillage Disease Classification

A comprehensive PyTorch-based plant disease classification system with path-preserving data policies, synthetic testing capabilities, and fusion of CNN and ANN approaches.

## Key Features

- **Path-Preserving Policy**: Never moves or renames dataset files
- **Manifest-Based Splits**: Uses `<root>/splits/<variant>/{train,val,test}.txt` with relative paths
- **Synthetic Mode**: Complete end-to-end testing without real datasets
- **CPU-Safe Defaults**: Works on any machine without GPU requirements
- **Dual Architecture**: CNN models (EfficientNet, MobileNet) + ANN models (MLP with handcrafted features)
- **Feature Fusion**: Ensemble predictions combining CNN and ANN outputs
- **Optional Segmentation**: MATLAB-based mask generation with mirrored directory structure
- **Comprehensive Evaluation**: GradCAM visualization, confusion matrices, and performance metrics

## Quick Start (Synthetic)

### Installation
```bash
pip install -r requirements.txt
```

### Complete Synthetic Pipeline (No Dataset Required)
```bash
make smoke
```

This runs the complete end-to-end pipeline including:
- **Synthetic Data Generation**: Creates 240 synthetic plant disease images across 6 classes
- **CNN Training**: EfficientNet-B0 training with early stopping and model exports (ONNX, TorchScript)
- **CNN Evaluation**: Test set evaluation with GradCAM visualizations for interpretability
- **Feature Extraction**: HSV statistics, HOG descriptors, and area ratios cached in NPZ format
- **ANN Training**: MLP neural network training on handcrafted features with curves and metrics
- **Model Fusion**: Ensemble predictions combining CNN and ANN outputs with comparison table
- **Artifact Generation**: Training curves, confusion matrices, classification reports, and model checkpoints

**Expected Runtime**: ~3 minutes on CPU

## Train on PlantVillage (Path-Preserving)

### 1. Prepare Your Dataset
Follow the path-preserving structure (never move or rename original files):
```
<root>/
├── <variant>/           # "color", "grayscale", or "segmented"
│   ├── class1/
│   │   ├── img001.jpg
│   │   └── ...
│   └── class2/
│       ├── img001.jpg
│       └── ...
├── splits/<variant>/    # Auto-generated manifests
│   ├── train.txt        # Relative paths: "class1/img001.jpg"
│   ├── val.txt
│   └── test.txt
└── masks/<variant>/     # Optional, mirrors original structure
    ├── class1/
    │   ├── img001.png
    │   └── ...
    └── class2/
        ├── img001.png
        └── ...
```

### 2. Configure Dataset
Update `config.yaml`:
```yaml
data:
  root: "/path/to/your/plantvillage"  # Point to your dataset
  variant: "color"                    # "color", "grayscale", or "segmented"
  subset_per_class: null              # Use full dataset (or set limit for dev)
```

### 3. Generate Manifests
Create train/val/test splits (only run once or when changing splits):
```bash
python src/data.py --regen-splits
```

### 4. Train Complete Pipeline
```bash
# Train CNN model
python src/train_cnn.py

# Evaluate CNN with GradCAM
python src/eval_cnn.py --gradcam 10 --tta

# Extract features and train ANN
python src/train_ann.py

# Fuse CNN and ANN predictions
python src/fuse.py
```

Or run everything at once:
```bash
make train && make eval && make train-ann && make fuse
```

## Configuration

Key settings in `config.yaml`:

- `data.root: ""` - Empty for synthetic mode, set path for real data
- `data.variant` - Dataset variant: "color", "grayscale", or "segmented"
- `data.subset_per_class` - Limit samples per class (for development)
- `train.model` - Model architecture: "efficientnet_b0" or "mobilenet_v3_small"
- `train.num_workers: 0` - CPU-safe data loading

## Available Models

### CNN Models
- **EfficientNet-B0**: Balanced accuracy and efficiency (default)
- **MobileNet-V3-Small**: Lightweight for mobile deployment and low-resource environments

### ANN Models  
- **MLP Neural Network**: 128→64→num_classes architecture with handcrafted features
- **Feature Extraction**: HSV statistics (6 features) + HOG descriptors + area ratios
- **Traditional ML**: HOG + SVM, Color Histogram + Texture (legacy support)

### Fusion Methods
- **Weighted Average**: Simple 0.5/0.5 combination of CNN and ANN softmax outputs
- **Performance Comparison**: Automatic generation of ANN vs CNN vs Fusion metrics table

## Makefile Targets

- `make smoke` - Complete synthetic pipeline (CNN + ANN + fusion)
- `make train` - Train CNN model
- `make eval` - Evaluate CNN model with GradCAM visualizations
- `make train-ann` - Extract features and train ANN model
- `make fuse` - Ensemble CNN and ANN predictions with comparison table

## Optional MATLAB Masks

Generate segmentation masks using MATLAB (requires MATLAB installation):

### 1. Generate Masks
```matlab
% In MATLAB command window
addpath('matlab_optional');
seg_masks('/path/to/dataset', '/path/to/dataset/masks', 'color');
```

This creates binary masks at `<root>/masks/<variant>/<class>/<basename>.png` using:
- HSV color space thresholding for green vegetation detection
- Lab color space refinement
- Morphological operations (opening, closing, hole filling)
- Largest connected component selection

### 2. Extract MATLAB Features (Optional)
```matlab
% Extract comprehensive handcrafted features
extract_feats('/path/to/dataset', 'features.mat', 'color');
```

Features include:
- Color histograms (RGB, 32 bins each)
- Color moments (mean, std for each channel)
- Texture features (GLCM statistics, Local Binary Patterns)
- Shape features (edge density, orientation histograms)
- Statistical features (intensity moments)

### 3. Enable Mask Usage
Update `config.yaml`:
```yaml
data:
  use_masks: true
  masks_subdir: "masks"
```

## Path-Preserving Policies

### Core Principles
1. **Never move or rename dataset files** - Preserves original organization
2. **Use manifest files** for train/val/test splits with relative paths
3. **Optional masks mirror original tree** structure
4. **No large data in repository** - Only code and small artifacts

### Benefits
- Easy integration with existing datasets
- Reproducible splits across experiments
- Clear separation of code and data
- Supports multiple dataset variants

## CI/CD Integration

GitHub Actions workflow automatically:
- Installs dependencies
- Runs synthetic smoke test
- Validates code quality with ruff and black

## Output Structure

```
checkpoints/          # Model weights (gitignored)
report_assets/        # Evaluation reports and visualizations
├── gradcam_*.png    # GradCAM visualizations
├── confusion_matrix.png
└── classification_report.txt
```

## Publishing Full Weights

For models >50MB, use GitHub Releases to distribute trained weights:

### 1. Create Release
```bash
# Tag your trained models
git tag v1.0.0-models
git push origin v1.0.0-models

# Create release with model files
gh release create v1.0.0-models \
  checkpoints/small_efficientnet_b0_state.pt \
  checkpoints/small_efficientnet_b0.onnx \
  checkpoints/small_efficientnet_b0.torchscript.pt \
  checkpoints/small_ann_state.pt \
  checkpoints/ann_scaler.pkl \
  --title "Trained Models v1.0.0" \
  --notes "Pre-trained models for PlantVillage disease classification"
```

### 2. Download Pre-trained Models
```bash
# Download from release
gh release download v1.0.0-models -D checkpoints/

# Or use direct URLs in your code
wget https://github.com/maadhusn/CN7023/releases/download/v1.0.0-models/small_efficientnet_b0_state.pt
```

## Low-Resource Tips

For training on limited hardware or small datasets:

### 1. Reduce Model Complexity
```yaml
train:
  model: "mobilenet_v3_small"  # Lighter than EfficientNet-B0
  batch_size: 16               # Reduce if memory limited
  mixed_precision: true        # Enable if GPU available
```

### 2. Limit Dataset Size
```yaml
data:
  subset_per_class: 100        # Use only 100 samples per class
  image_size: 128              # Smaller input resolution
```

### 3. Reduce Training Time
```yaml
train:
  epochs: 10                   # Fewer epochs for quick experiments
  early_stop_patience: 3       # Stop early if no improvement
```

### 4. CPU-Only Training
```yaml
train:
  num_workers: 0               # Avoid multiprocessing issues
  mixed_precision: false       # CPU doesn't support mixed precision
```

**Memory Usage**: MobileNet-V3-Small with 128px images uses ~2GB RAM vs ~4GB for EfficientNet-B0 with 192px images.

## Development

### Code Quality
```bash
ruff check .          # Linting
black .               # Formatting
```

### Testing
```bash
make smoke            # Quick synthetic test
python -m pytest     # Full test suite (if available)
```

## License

MIT License - see LICENSE file for details.
