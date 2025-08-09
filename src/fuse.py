"""Fusion of CNN and ANN predictions for improved performance."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from data import create_data_loaders, load_config
from eval_cnn import load_trained_model
from features import load_features
from models.ann import HOGClassifier
from utils import calculate_metrics, get_device, plot_confusion_matrix, save_metrics, set_seed


def load_ann_model(model_path: str) -> HOGClassifier:
    """Load trained ANN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ANN model not found: {model_path}")
    
    classifier = HOGClassifier()
    classifier.load(model_path)
    return classifier


def get_cnn_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """Get CNN predictions and probabilities."""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="CNN predictions"):
            data = data.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)


def get_ann_predictions(
    classifier: HOGClassifier,
    features: np.ndarray,
    labels: np.ndarray
) -> tuple:
    """Get ANN predictions and probabilities."""
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    
    return predictions, probabilities, labels


class EnsembleFusion:
    """Ensemble fusion of CNN and ANN predictions."""
    
    def __init__(self, fusion_method: str = 'weighted_average'):
        self.fusion_method = fusion_method
        self.cnn_weight = 0.7  # Default weight for CNN
        self.ann_weight = 0.3  # Default weight for ANN
        self.is_fitted = False
    
    def fit(
        self,
        cnn_probs: np.ndarray,
        ann_probs: np.ndarray,
        y_true: np.ndarray
    ):
        """Fit fusion weights based on validation performance."""
        if self.fusion_method == 'learned_weights':
            best_accuracy = 0
            best_cnn_weight = 0.5
            
            for cnn_w in np.arange(0.1, 1.0, 0.1):
                ann_w = 1.0 - cnn_w
                fused_probs = cnn_w * cnn_probs + ann_w * ann_probs
                predictions = np.argmax(fused_probs, axis=1)
                accuracy = accuracy_score(y_true, predictions)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_cnn_weight = cnn_w
            
            self.cnn_weight = best_cnn_weight
            self.ann_weight = 1.0 - best_cnn_weight
            
            print(f"Learned weights - CNN: {self.cnn_weight:.3f}, ANN: {self.ann_weight:.3f}")
            print(f"Validation accuracy with learned weights: {best_accuracy:.4f}")
        
        self.is_fitted = True
    
    def predict(
        self,
        cnn_probs: np.ndarray,
        ann_probs: np.ndarray
    ) -> tuple:
        """Fuse predictions from CNN and ANN."""
        if not self.is_fitted and self.fusion_method == 'learned_weights':
            raise ValueError("Fusion model must be fitted first for learned weights")
        
        if self.fusion_method == 'weighted_average' or self.fusion_method == 'learned_weights':
            fused_probs = self.cnn_weight * cnn_probs + self.ann_weight * ann_probs
            predictions = np.argmax(fused_probs, axis=1)
            
        elif self.fusion_method == 'majority_vote':
            cnn_preds = np.argmax(cnn_probs, axis=1)
            ann_preds = np.argmax(ann_probs, axis=1)
            
            predictions = []
            fused_probs = []
            
            for i in range(len(cnn_preds)):
                if cnn_preds[i] == ann_preds[i]:
                    predictions.append(cnn_preds[i])
                    prob = (cnn_probs[i] + ann_probs[i]) / 2
                else:
                    predictions.append(cnn_preds[i])
                    prob = cnn_probs[i]
                
                fused_probs.append(prob)
            
            predictions = np.array(predictions)
            fused_probs = np.array(fused_probs)
        
        elif self.fusion_method == 'max_confidence':
            cnn_max_probs = np.max(cnn_probs, axis=1)
            ann_max_probs = np.max(ann_probs, axis=1)
            
            predictions = []
            fused_probs = []
            
            for i in range(len(cnn_max_probs)):
                if cnn_max_probs[i] >= ann_max_probs[i]:
                    predictions.append(np.argmax(cnn_probs[i]))
                    fused_probs.append(cnn_probs[i])
                else:
                    predictions.append(np.argmax(ann_probs[i]))
                    fused_probs.append(ann_probs[i])
            
            predictions = np.array(predictions)
            fused_probs = np.array(fused_probs)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return predictions, fused_probs


def evaluate_fusion_methods(
    cnn_probs: np.ndarray,
    ann_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list
) -> dict:
    """Evaluate different fusion methods."""
    fusion_methods = ['weighted_average', 'learned_weights', 'majority_vote', 'max_confidence']
    results = {}
    
    for method in fusion_methods:
        print(f"\nEvaluating fusion method: {method}")
        
        fusion = EnsembleFusion(fusion_method=method)
        
        if method == 'learned_weights':
            fusion.fit(cnn_probs, ann_probs, y_true)
        else:
            fusion.is_fitted = True
        
        predictions, fused_probs = fusion.predict(cnn_probs, ann_probs)
        
        metrics = calculate_metrics(y_true, predictions, class_names)
        
        results[method] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'predictions': predictions.tolist(),
            'probabilities': fused_probs.tolist(),
            'fusion_weights': {
                'cnn': fusion.cnn_weight,
                'ann': fusion.ann_weight
            }
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    
    return results


def main():
    """Main fusion function."""
    config = load_config()
    
    set_seed(42)
    
    device = get_device()
    
    print("Loading models and data...")
    
    model_name = config['train']['model']
    checkpoint_dir = Path("checkpoints")
    
    cnn_model_path = checkpoint_dir / f"{model_name}_best.pth"
    if not cnn_model_path.exists():
        cnn_model_path = checkpoint_dir / f"{model_name}_final.pth"
    
    if not cnn_model_path.exists():
        print("CNN model not found. Please train CNN first.")
        return
    
    cnn_model, model_config, class_names = load_trained_model(str(cnn_model_path), device)
    print(f"Loaded CNN model: {cnn_model_path}")
    
    ann_model_path = "checkpoints/ann_best_model.pkl"
    if not os.path.exists(ann_model_path):
        print("ANN model not found. Please train ANN first.")
        return
    
    ann_model = load_ann_model(ann_model_path)
    print(f"Loaded ANN model: {ann_model_path}")
    
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    try:
        features, labels = load_features()
    except FileNotFoundError:
        print("Features not found. Please run feature extraction first.")
        return
    
    print("\nGetting validation predictions...")
    
    cnn_val_preds, cnn_val_probs, val_targets = get_cnn_predictions(
        cnn_model, val_loader, device
    )
    
    ann_val_preds, ann_val_probs, _ = get_ann_predictions(
        ann_model, features['val'], labels['val']
    )
    
    cnn_val_acc = accuracy_score(val_targets, cnn_val_preds)
    ann_val_acc = accuracy_score(val_targets, ann_val_preds)
    
    print(f"CNN validation accuracy: {cnn_val_acc:.4f}")
    print(f"ANN validation accuracy: {ann_val_acc:.4f}")
    
    print("\nEvaluating fusion methods on validation set...")
    val_fusion_results = evaluate_fusion_methods(
        cnn_val_probs, ann_val_probs, val_targets, class_names
    )
    
    best_method = max(val_fusion_results.keys(), 
                     key=lambda k: val_fusion_results[k]['accuracy'])
    
    print(f"\nBest fusion method: {best_method}")
    print(f"Best validation accuracy: {val_fusion_results[best_method]['accuracy']:.4f}")
    
    print(f"\nTesting with {best_method} on test set...")
    
    cnn_test_preds, cnn_test_probs, test_targets = get_cnn_predictions(
        cnn_model, test_loader, device
    )
    
    ann_test_preds, ann_test_probs, _ = get_ann_predictions(
        ann_model, features['test'], labels['test']
    )
    
    best_fusion = EnsembleFusion(fusion_method=best_method)
    
    if best_method == 'learned_weights':
        best_fusion.fit(cnn_val_probs, ann_val_probs, val_targets)
    else:
        best_fusion.is_fitted = True
    
    fused_test_preds, fused_test_probs = best_fusion.predict(
        cnn_test_probs, ann_test_probs
    )
    
    test_metrics = calculate_metrics(test_targets, fused_test_preds, class_names)
    
    cnn_test_acc = accuracy_score(test_targets, cnn_test_preds)
    ann_test_acc = accuracy_score(test_targets, ann_test_preds)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
    print(f"ANN Test Accuracy: {ann_test_acc:.4f}")
    print(f"Fused Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Improvement over CNN: {test_metrics['accuracy'] - cnn_test_acc:+.4f}")
    print(f"Improvement over ANN: {test_metrics['accuracy'] - ann_test_acc:+.4f}")
    
    os.makedirs("report_assets", exist_ok=True)
    
    fusion_results = {
        'validation_results': val_fusion_results,
        'best_method': best_method,
        'test_results': {
            'cnn_accuracy': cnn_test_acc,
            'ann_accuracy': ann_test_acc,
            'fused_accuracy': test_metrics['accuracy'],
            'fused_metrics': test_metrics,
            'fusion_weights': {
                'cnn': best_fusion.cnn_weight,
                'ann': best_fusion.ann_weight
            }
        }
    }
    
    save_metrics(fusion_results, "report_assets/fusion_results.json")
    
    cm = np.array(test_metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names,
        save_path="report_assets/fusion_confusion_matrix.png"
    )
    
    print("\nFusion evaluation completed!")
    print("Check report_assets/ for detailed results.")


if __name__ == "__main__":
    main()
