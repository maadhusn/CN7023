"""ANN model using handcrafted features for plant disease classification."""

import pickle
from typing import Optional, Tuple

import cv2
import numpy as np
import torch.nn as nn
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


class HOGFeatureExtractor:
    """HOG feature extractor for images."""

    def __init__(
        self,
        pixels_per_cell: Tuple[int, int] = (16, 16),
        cells_per_block: Tuple[int, int] = (2, 2),
        orientations: int = 9,
        block_norm: str = 'L2-Hys'
    ):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations
        self.block_norm = block_norm

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from an image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            feature_vector=True
        )

        return features

    def extract_features_batch(self, images: list) -> np.ndarray:
        """Extract HOG features from a batch of images."""
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)

        return np.array(features_list)


class ColorHistogramExtractor:
    """Color histogram feature extractor."""

    def __init__(self, bins: int = 32):
        self.bins = bins

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        hist_features = []

        if len(image.shape) == 3:
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([image], [i], None, [self.bins], [0, 256])
                hist_features.extend(hist.flatten())
        else:
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])
            hist_features.extend(hist.flatten())

        return np.array(hist_features)


class TextureFeatureExtractor:
    """Texture feature extractor using Local Binary Patterns."""

    def __init__(self, radius: int = 3, n_points: int = 24):
        self.radius = radius
        self.n_points = n_points

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract LBP texture features."""
        from skimage.feature import local_binary_pattern

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')

        hist, _ = np.histogram(lbp.ravel(), bins=self.n_points + 2, range=(0, self.n_points + 2))

        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)

        return hist


class HOGClassifier:
    """Traditional ML classifier using HOG and other handcrafted features."""

    def __init__(
        self,
        classifier_type: str = 'random_forest',
        hog_params: Optional[dict] = None,
        use_color_hist: bool = True,
        use_texture: bool = True
    ):
        self.classifier_type = classifier_type
        self.use_color_hist = use_color_hist
        self.use_texture = use_texture

        hog_params = hog_params or {}
        self.hog_extractor = HOGFeatureExtractor(**hog_params)

        if use_color_hist:
            self.color_extractor = ColorHistogramExtractor()

        if use_texture:
            self.texture_extractor = TextureFeatureExtractor()

        self.classifier = self._create_classifier()
        self.is_fitted = False

    def _create_classifier(self):
        """Create the specified classifier."""
        if self.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        elif self.classifier_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all features from an image."""
        features = []

        hog_features = self.hog_extractor.extract_features(image)
        features.extend(hog_features)

        if self.use_color_hist:
            color_features = self.color_extractor.extract_features(image)
            features.extend(color_features)

        if self.use_texture:
            texture_features = self.texture_extractor.extract_features(image)
            features.extend(texture_features)

        return np.array(features)

    def extract_features_batch(self, images: list) -> np.ndarray:
        """Extract features from a batch of images."""
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)

        return np.array(features_list)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        if len(X.shape) == 4:  # Batch of images
            X_features = self.extract_features_batch(X)
        else:  # Already extracted features
            X_features = X

        self.classifier.fit(X_features, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if len(X.shape) == 4:  # Batch of images
            X_features = self.extract_features_batch(X)
        else:  # Already extracted features
            X_features = X

        return self.classifier.predict(X_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if len(X.shape) == 4:  # Batch of images
            X_features = self.extract_features_batch(X)
        else:  # Already extracted features
            X_features = X

        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_features)
        else:
            predictions = self.classifier.predict(X_features)
            n_classes = len(np.unique(predictions))
            probas = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0
            return probas

    def evaluate(self, X: np.ndarray, y: np.ndarray, class_names: list) -> dict:
        """Evaluate the classifier."""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        report = classification_report(
            y, predictions, target_names=class_names, output_dict=True
        )

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions.tolist()
        }

    def save(self, filepath: str):
        """Save the trained classifier."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted classifier")

        model_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'use_color_hist': self.use_color_hist,
            'use_texture': self.use_texture,
            'hog_extractor': self.hog_extractor,
        }

        if self.use_color_hist:
            model_data['color_extractor'] = self.color_extractor

        if self.use_texture:
            model_data['texture_extractor'] = self.texture_extractor

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str):
        """Load a trained classifier."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.use_color_hist = model_data['use_color_hist']
        self.use_texture = model_data['use_texture']
        self.hog_extractor = model_data['hog_extractor']

        if self.use_color_hist:
            self.color_extractor = model_data['color_extractor']

        if self.use_texture:
            self.texture_extractor = model_data['texture_extractor']

        self.is_fitted = True


class MLP(nn.Module):
    """Multi-Layer Perceptron for feature-based classification."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
