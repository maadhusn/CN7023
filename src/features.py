"""Feature extraction utilities for traditional ML approaches."""

import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import create_data_loaders, load_config
from utils import set_seed


def extract_hsv_features(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Extract HSV mean and std features (3x2 = 6 features)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if mask is not None:
        hsv_masked = hsv[mask > 0]
        if len(hsv_masked) == 0:
            return np.zeros(6)
        mean_vals = np.mean(hsv_masked, axis=0)
        std_vals = np.std(hsv_masked, axis=0)
    else:
        mean_vals = np.mean(hsv.reshape(-1, 3), axis=0)
        std_vals = np.std(hsv.reshape(-1, 3), axis=0)

    return np.concatenate([mean_vals, std_vals])


def extract_hog_features(image: np.ndarray, config: dict) -> np.ndarray:
    """Extract HOG features (downsampled)."""
    from skimage.feature import hog

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    ann_config = config["ann"]

    features = hog(
        gray,
        orientations=ann_config["hog_bins"],
        pixels_per_cell=(
            ann_config["hog_pixels_per_cell"],
            ann_config["hog_pixels_per_cell"],
        ),
        cells_per_block=(
            ann_config["hog_cells_per_block"],
            ann_config["hog_cells_per_block"],
        ),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )

    return features


def extract_area_ratio(image: np.ndarray, mask: np.ndarray = None) -> float:
    """Extract area ratio feature."""
    if mask is not None:
        foreground_area = np.sum(mask > 0)
        total_area = mask.size
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_area = np.sum(binary > 0)
        total_area = binary.size

    return foreground_area / total_area if total_area > 0 else 0.0


def extract_features_from_image(
    image_path: str, config: dict, mask_path: str = None
) -> np.ndarray:
    """Extract all features from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    features = []

    hsv_features = extract_hsv_features(image, mask)
    features.extend(hsv_features)

    hog_features = extract_hog_features(image, config)
    features.extend(hog_features)

    area_ratio = extract_area_ratio(image, mask)
    features.append(area_ratio)

    return np.array(features)


def extract_features_from_loader(
    data_loader: torch.utils.data.DataLoader,
    config: dict,
    split_name: str,
    cache_dir: str = "report_assets/features",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features from a data loader and cache to NPZ."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"features_{split_name}.npz")

    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data["features"], data["labels"], data["paths"].tolist()

    all_features = []
    all_labels = []
    all_paths = []

    print(f"Extracting features for {split_name} split...")

    for batch_idx, (data, targets) in enumerate(
        tqdm(data_loader, desc=f"Extracting {split_name} features")
    ):
        for i in range(data.size(0)):
            image_tensor = data[i]
            image_np = image_tensor.numpy().transpose(1, 2, 0)

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)

            synthetic_path = f"synthetic_{split_name}_{batch_idx}_{i}.jpg"

            hsv_features = extract_hsv_features(image_np)
            hog_features = extract_hog_features(image_np, config)
            area_ratio = extract_area_ratio(image_np)

            sample_features = np.concatenate([hsv_features, hog_features, [area_ratio]])

            all_features.append(sample_features)
            all_labels.append(targets[i].item())
            all_paths.append(synthetic_path)

    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    paths_array = np.array(all_paths)

    np.savez(
        cache_file, features=features_array, labels=labels_array, paths=paths_array
    )
    print(f"Cached features to {cache_file}")

    return features_array, labels_array, all_paths


def create_feature_extractors(config: dict) -> dict:
    """Create feature extractors based on configuration (legacy function)."""
    from models.ann import (
        ColorHistogramExtractor,
        HOGFeatureExtractor,
        TextureFeatureExtractor,
    )

    ann_config = config["ann"]

    extractors = {
        "hog": HOGFeatureExtractor(
            pixels_per_cell=(
                ann_config["hog_pixels_per_cell"],
                ann_config["hog_pixels_per_cell"],
            ),
            cells_per_block=(
                ann_config["hog_cells_per_block"],
                ann_config["hog_cells_per_block"],
            ),
            orientations=ann_config["hog_bins"],
        ),
        "color_hist": ColorHistogramExtractor(bins=32),
        "texture": TextureFeatureExtractor(radius=3, n_points=24),
    }

    return extractors


def reduce_dimensionality(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: float = 0.95,
    save_dir: str = "checkpoints",
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

    with open(os.path.join(save_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(save_dir, "feature_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    return X_train_pca, X_val_pca, X_test_pca


def save_features(features: dict, labels: dict, save_dir: str = "checkpoints"):
    """Save extracted features to disk."""
    os.makedirs(save_dir, exist_ok=True)

    features_path = os.path.join(save_dir, "extracted_features.pkl")

    data = {"features": features, "labels": labels}

    with open(features_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Features saved to {features_path}")


def load_features(save_dir: str = "checkpoints") -> Tuple[dict, dict]:
    """Load extracted features from disk."""
    features_path = os.path.join(save_dir, "extracted_features.pkl")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    return data["features"], data["labels"]


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_dir: str = "report_assets",
):
    """Analyze feature importance using various methods."""
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

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
    ax1.set_xlabel("Importance")
    ax1.set_title("Random Forest Feature Importance (Top 20)")

    top_indices_mi = np.argsort(mi_scores)[-20:]
    ax2.barh(range(len(top_indices_mi)), mi_scores[top_indices_mi])
    ax2.set_yticks(range(len(top_indices_mi)))
    ax2.set_yticklabels([f"Feature {i}" for i in top_indices_mi])
    ax2.set_xlabel("Mutual Information Score")
    ax2.set_title("Mutual Information Feature Importance (Top 20)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"Feature importance analysis saved to {save_dir}/feature_importance.png")


def main():
    """Main feature extraction function."""

    config = load_config()

    set_seed(42)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)

    print("Creating feature extractors...")
    extractors = create_feature_extractors(config)

    print("Extracting features from training set...")
    X_train, y_train = extract_features_from_loader(
        train_loader,
        extractors,
        max_samples=config["data"].get("subset_per_class", None),
    )

    print("Extracting features from validation set...")
    X_val, y_val = extract_features_from_loader(
        val_loader, extractors, max_samples=config["data"].get("subset_per_class", None)
    )

    print("Extracting features from test set...")
    X_test, y_test = extract_features_from_loader(
        test_loader,
        extractors,
        max_samples=config["data"].get("subset_per_class", None),
    )

    print("Feature extraction completed!")
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")

    print("Applying dimensionality reduction...")
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_dimensionality(
        X_train, X_val, X_test
    )

    features = {"train": X_train_reduced, "val": X_val_reduced, "test": X_test_reduced}

    labels = {"train": y_train, "val": y_val, "test": y_test}

    save_features(features, labels)

    print("Analyzing feature importance...")
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    analyze_feature_importance(X_train, y_train, feature_names)

    print("Feature extraction and analysis completed!")


if __name__ == "__main__":
    main()
