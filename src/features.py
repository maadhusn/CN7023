"""Feature extraction utilities for traditional ML approaches."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import create_data_loaders, load_config
from models.ann import HOGFeatureExtractor, ColorHistogramExtractor, TextureFeatureExtractor
from utils import set_seed


def extract_features_from_loader(
    data_loader: torch.utils.data.DataLoader,
    feature_extractors: dict,
    max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a data loader."""
    all_features = []
    all_labels = []
    
    samples_processed = 0
    
    for batch_idx, (data, targets) in enumerate(tqdm(data_loader, desc="Extracting features")):
        for i in range(data.size(0)):
            if max_samples and samples_processed >= max_samples:
                break
            
            image_tensor = data[i]
            image_np = image_tensor.numpy().transpose(1, 2, 0)
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            
            sample_features = []
            
            for name, extractor in feature_extractors.items():
                features = extractor.extract_features(image_np)
                sample_features.extend(features)
            
            all_features.append(sample_features)
            all_labels.append(targets[i].item())
            samples_processed += 1
        
        if max_samples and samples_processed >= max_samples:
            break
    
    return np.array(all_features), np.array(all_labels)


def create_feature_extractors(config: dict) -> dict:
    """Create feature extractors based on configuration."""
    ann_config = config['ann']
    
    extractors = {
        'hog': HOGFeatureExtractor(
            pixels_per_cell=(ann_config['hog_pixels_per_cell'], ann_config['hog_pixels_per_cell']),
            cells_per_block=(ann_config['hog_cells_per_block'], ann_config['hog_cells_per_block']),
            orientations=ann_config['hog_bins']
        ),
        'color_hist': ColorHistogramExtractor(bins=32),
        'texture': TextureFeatureExtractor(radius=3, n_points=24)
    }
    
    return extractors


def reduce_dimensionality(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: float = 0.95,
    save_dir: str = "checkpoints"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply PCA for dimensionality reduction."""
    print(f"Original feature dimension: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Reduced feature dimension: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "feature_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(save_dir, "feature_pca.pkl"), 'wb') as f:
        pickle.dump(pca, f)
    
    return X_train_pca, X_val_pca, X_test_pca


def save_features(
    features: dict,
    labels: dict,
    save_dir: str = "checkpoints"
):
    """Save extracted features to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    features_path = os.path.join(save_dir, "extracted_features.pkl")
    
    data = {
        'features': features,
        'labels': labels
    }
    
    with open(features_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Features saved to {features_path}")


def load_features(save_dir: str = "checkpoints") -> Tuple[dict, dict]:
    """Load extracted features from disk."""
    features_path = os.path.join(save_dir, "extracted_features.pkl")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['features'], data['labels']


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_dir: str = "report_assets"
):
    """Analyze feature importance using various methods."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    top_indices_rf = np.argsort(rf_importance)[-20:]
    ax1.barh(range(len(top_indices_rf)), rf_importance[top_indices_rf])
    ax1.set_yticks(range(len(top_indices_rf)))
    ax1.set_yticklabels([f"Feature {i}" for i in top_indices_rf])
    ax1.set_xlabel('Importance')
    ax1.set_title('Random Forest Feature Importance (Top 20)')
    
    top_indices_mi = np.argsort(mi_scores)[-20:]
    ax2.barh(range(len(top_indices_mi)), mi_scores[top_indices_mi])
    ax2.set_yticks(range(len(top_indices_mi)))
    ax2.set_yticklabels([f"Feature {i}" for i in top_indices_mi])
    ax2.set_xlabel('Mutual Information Score')
    ax2.set_title('Mutual Information Feature Importance (Top 20)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance analysis saved to {save_dir}/feature_importance.png")


def main():
    """Main feature extraction function."""
    import torch
    
    config = load_config()
    
    set_seed(42)
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print("Creating feature extractors...")
    extractors = create_feature_extractors(config)
    
    print("Extracting features from training set...")
    X_train, y_train = extract_features_from_loader(
        train_loader, extractors, max_samples=config['data'].get('subset_per_class', None)
    )
    
    print("Extracting features from validation set...")
    X_val, y_val = extract_features_from_loader(
        val_loader, extractors, max_samples=config['data'].get('subset_per_class', None)
    )
    
    print("Extracting features from test set...")
    X_test, y_test = extract_features_from_loader(
        test_loader, extractors, max_samples=config['data'].get('subset_per_class', None)
    )
    
    print(f"Feature extraction completed!")
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    print("Applying dimensionality reduction...")
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_dimensionality(
        X_train, X_val, X_test
    )
    
    features = {
        'train': X_train_reduced,
        'val': X_val_reduced,
        'test': X_test_reduced
    }
    
    labels = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    save_features(features, labels)
    
    print("Analyzing feature importance...")
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    analyze_feature_importance(X_train, y_train, feature_names)
    
    print("Feature extraction and analysis completed!")


if __name__ == "__main__":
    main()
