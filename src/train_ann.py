"""Train ANN models using traditional ML with handcrafted features."""

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from data import load_config
from features import load_features
from models.ann import HOGClassifier
from utils import calculate_metrics, plot_confusion_matrix, save_metrics, set_seed


def train_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    class_names: list
) -> HOGClassifier:
    """Train ANN model with handcrafted features."""
    ann_config = config['ann']
    
    classifier = HOGClassifier(
        classifier_type='random_forest',
        hog_params={
            'pixels_per_cell': (ann_config['hog_pixels_per_cell'], ann_config['hog_pixels_per_cell']),
            'cells_per_block': (ann_config['hog_cells_per_block'], ann_config['hog_cells_per_block']),
            'orientations': ann_config['hog_bins']
        },
        use_color_hist=True,
        use_texture=True
    )
    
    print("Training Random Forest classifier...")
    classifier.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    val_metrics = classifier.evaluate(X_val, y_val, class_names)
    
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    return classifier


def compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    class_names: list
) -> dict:
    """Compare different traditional ML classifiers."""
    ann_config = config['ann']
    
    classifiers = {
        'Random Forest': HOGClassifier(
            classifier_type='random_forest',
            hog_params={
                'pixels_per_cell': (ann_config['hog_pixels_per_cell'], ann_config['hog_pixels_per_cell']),
                'cells_per_block': (ann_config['hog_cells_per_block'], ann_config['hog_cells_per_block']),
                'orientations': ann_config['hog_bins']
            }
        ),
        'SVM': HOGClassifier(
            classifier_type='svm',
            hog_params={
                'pixels_per_cell': (ann_config['hog_pixels_per_cell'], ann_config['hog_pixels_per_cell']),
                'cells_per_block': (ann_config['hog_cells_per_block'], ann_config['hog_cells_per_block']),
                'orientations': ann_config['hog_bins']
            }
        ),
        'Logistic Regression': HOGClassifier(
            classifier_type='logistic',
            hog_params={
                'pixels_per_cell': (ann_config['hog_pixels_per_cell'], ann_config['hog_pixels_per_cell']),
                'cells_per_block': (ann_config['hog_cells_per_block'], ann_config['hog_cells_per_block']),
                'orientations': ann_config['hog_bins']
            }
        )
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        classifier.fit(X_train, y_train)
        
        val_metrics = classifier.evaluate(X_val, y_val, class_names)
        results[name] = {
            'classifier': classifier,
            'accuracy': val_metrics['accuracy'],
            'metrics': val_metrics
        }
        
        print(f"{name} Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    return results


def main():
    """Main ANN training function."""
    config = load_config()
    
    set_seed(42)
    
    print("Loading extracted features...")
    try:
        features, labels = load_features()
    except FileNotFoundError:
        print("Features not found. Please run feature extraction first:")
        print("python src/features.py")
        return
    
    X_train = features['train']
    y_train = labels['train']
    X_val = features['val']
    y_val = labels['val']
    X_test = features['test']
    y_test = labels['test']
    
    print(f"Loaded features:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    class_names = [f"class_{i}" for i in range(len(np.unique(y_train)))]
    
    print("\nComparing different classifiers...")
    results = compare_classifiers(X_train, y_train, X_val, y_val, config, class_names)
    
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_classifier = results[best_name]['classifier']
    
    print(f"\nBest classifier: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
    
    print(f"\nEvaluating {best_name} on test set...")
    test_metrics = best_classifier.evaluate(X_test, y_test, class_names)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/ann_best_model.pkl"
    best_classifier.save(model_path)
    print(f"Best model saved to {model_path}")
    
    os.makedirs("report_assets", exist_ok=True)
    
    save_metrics(test_metrics, "report_assets/ann_test_metrics.json")
    
    cm = np.array([[test_metrics['classification_report'][cls]['support'] for cls in class_names]])
    y_pred = test_metrics['predictions']
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plot_confusion_matrix(
        cm, class_names,
        save_path="report_assets/ann_confusion_matrix.png"
    )
    
    comparison_results = {}
    for name, result in results.items():
        comparison_results[name] = {
            'accuracy': result['accuracy'],
            'classification_report': result['metrics']['classification_report']
        }
    
    save_metrics(comparison_results, "report_assets/ann_comparison.json")
    
    print("\nANN training completed!")
    print("Check report_assets/ for detailed results.")


if __name__ == "__main__":
    main()
