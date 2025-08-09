# CN7023 - PlantVillage Disease Classification

A PyTorch-based plant disease classification system with path-preserving data policies and synthetic testing capabilities.

## Key Features

- **Path-Preserving Policy**: Never moves or renames dataset files
- **Manifest-Based Splits**: Uses `<root>/splits/<variant>/{train,val,test}.txt` with relative paths
- **Synthetic Mode**: Complete end-to-end testing without real datasets
- **CPU-Safe Defaults**: Works on any machine without GPU requirements
- **Flexible Architecture**: Supports both CNN (EfficientNet, MobileNet) and ANN (HOG) models
- **Optional Segmentation**: Mask support with mirrored directory structure

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Synthetic Smoke Test (No Dataset Required)
```bash
make smoke
```

This runs a complete synthetic demo that:
- Creates synthetic plant disease images
- Tests model training and inference
- Validates the entire pipeline
- Generates output artifacts

### Using Real Data

1. **Prepare your dataset** following the path-preserving structure:
```
<root>/
├── <variant>/           # "color", "grayscale", or "segmented"
│   ├── class1/
│   │   ├── img001.jpg
│   │   └── ...
│   └── class2/
│       ├── img001.jpg
│       └── ...
├── splits/<variant>/
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

2. **Update config.yaml**:
```yaml
data:
  root: "/path/to/your/dataset"  # Set to your dataset path
  variant: "color"               # or "grayscale", "segmented"
  subset_per_class: null         # Use full dataset
```

3. **Train and evaluate**:
```bash
make train
make eval
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
- **EfficientNet-B0**: Balanced accuracy and efficiency
- **MobileNet-V3-Small**: Lightweight for mobile deployment

### ANN Models
- **HOG + SVM**: Traditional computer vision approach
- **Color Histogram + Texture**: Handcrafted feature extraction

## Makefile Targets

- `make smoke` - Synthetic end-to-end test
- `make train` - Train CNN model
- `make eval` - Evaluate trained model with GradCAM
- `make features` - Extract features for ANN training
- `make train-ann` - Train ANN model
- `make fuse` - Ensemble CNN and ANN predictions

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

## Publishing Large Models

For models >50MB, use GitHub Releases:
```bash
# Tag and create release
git tag v1.0.0
git push origin v1.0.0

# Upload large model files to release assets
gh release create v1.0.0 checkpoints/best_model.pth
```

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
