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


def load_mlp_model(device: torch.device) -> torch.nn.Module:
    """Load trained MLP model."""
    import pickle
    from models.ann import MLP
    
    state_dict_path = "checkpoints/small_ann_state.pt"
    scaler_path = "checkpoints/ann_scaler.pkl"
    
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"MLP state dict not found: {state_dict_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open("report_assets/ann_metrics.json", 'r') as f:
        ann_metrics = json.load(f)
    
    input_dim = ann_metrics['feature_dim']
    num_classes = ann_metrics['num_classes']
    
    model = MLP(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    
    return model, scaler


def get_mlp_predictions(
    model: torch.nn.Module,
    scaler,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device
) -> tuple:
    """Get MLP predictions and probabilities."""
    features_scaled = scaler.transform(features)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(features_scaled),
        torch.LongTensor(labels)
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            
            outputs = model(batch_features)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(batch_labels.numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)


def main():
    """Main fusion function."""
    config = load_config()
    
    set_seed(42)
    device = get_device()
    
    print("Loading models and data...")
    
    model_name = config['train']['model']
    checkpoint_dir = Path("checkpoints")
    
    state_dict_path = checkpoint_dir / f"small_{model_name}_state.pt"
    if not state_dict_path.exists():
        print("CNN model not found. Please train CNN first.")
        return
    
    from models.cnn import create_model
    num_classes = len(config.get('classes', ['class_0', 'class_1', 'class_2', 'class_3']))
    cnn_model = create_model(model_name, num_classes, pretrained=False)
    cnn_model.load_state_dict(torch.load(state_dict_path, map_location=device))
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    print(f"Loaded CNN model: {state_dict_path}")
    
    try:
        mlp_model, scaler = load_mlp_model(device)
        print("Loaded MLP model: checkpoints/small_ann_state.pt")
    except FileNotFoundError as e:
        print(f"MLP model not found: {e}")
        return
    
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    from features import extract_features_from_loader
    print("Extracting features for fusion...")
    X_test, y_test, _ = extract_features_from_loader(test_loader, config, "test")
    
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    print("\nGetting test predictions...")
    
    cnn_test_preds, cnn_test_probs, test_targets = get_cnn_predictions(
        cnn_model, test_loader, device
    )
    
    ann_test_preds, ann_test_probs, _ = get_mlp_predictions(
        mlp_model, scaler, X_test, y_test, device
    )
    
    cnn_test_acc = accuracy_score(test_targets, cnn_test_preds)
    ann_test_acc = accuracy_score(test_targets, ann_test_preds)
    
    fused_probs = 0.5 * cnn_test_probs + 0.5 * ann_test_probs
    fused_preds = np.argmax(fused_probs, axis=1)
    fused_test_acc = accuracy_score(test_targets, fused_preds)
    
    print("\n" + "="*60)
    print("COMPARISON TABLE: ANN vs CNN vs Fusion")
    print("="*60)
    print(f"{'Method':<15} {'Accuracy':<10} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'ANN':<15} {ann_test_acc:<10.4f} {'-':<12}")
    print(f"{'CNN':<15} {cnn_test_acc:<10.4f} {cnn_test_acc - ann_test_acc:+.4f}")
    print(f"{'Fusion':<15} {fused_test_acc:<10.4f} {fused_test_acc - max(ann_test_acc, cnn_test_acc):+.4f}")
    print("="*60)
    
    os.makedirs("report_assets", exist_ok=True)
    
    fusion_metrics = {
        'ann_accuracy': ann_test_acc,
        'cnn_accuracy': cnn_test_acc,
        'fusion_accuracy': fused_test_acc,
        'improvement_over_ann': fused_test_acc - ann_test_acc,
        'improvement_over_cnn': fused_test_acc - cnn_test_acc,
        'improvement_over_best': fused_test_acc - max(ann_test_acc, cnn_test_acc),
        'fusion_method': 'weighted_average',
        'fusion_weights': {'cnn': 0.5, 'ann': 0.5}
    }
    
    with open("report_assets/fusion_metrics.json", 'w') as f:
        json.dump(fusion_metrics, f, indent=2)
    
    print(f"\nFusion metrics saved to report_assets/fusion_metrics.json")
    print("\nFusion evaluation completed!")


if __name__ == "__main__":
    main()
