"""Evaluate CNN models and generate comprehensive reports."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import create_data_loaders, load_config
from models.cnn import create_model
from utils import (
    calculate_metrics,
    get_device,
    plot_confusion_matrix,
    save_metrics,
    set_seed,
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_predictions, class_names)
    
    metrics['probabilities'] = np.array(all_probabilities).tolist()
    metrics['predictions'] = all_predictions
    metrics['targets'] = all_targets
    
    return metrics


def load_trained_model(model_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', load_config())
    classes = checkpoint.get('classes', ['class_0', 'class_1', 'class_2', 'class_3'])
    
    model_name = config['train']['model']
    num_classes = len(classes)
    model = create_model(model_name, num_classes, pretrained=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config, classes


def generate_evaluation_report(metrics: dict, save_dir: str = "report_assets"):
    """Generate comprehensive evaluation report."""
    os.makedirs(save_dir, exist_ok=True)
    
    save_metrics(metrics, os.path.join(save_dir, "evaluation_metrics.json"))
    
    cm = np.array(metrics['confusion_matrix'])
    class_names = list(metrics['classification_report'].keys())[:-3]  # Remove avg keys
    
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png")
    )
    
    report_path = os.path.join(save_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Plant Disease Classification - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write("-" * 30 + "\n")
        
        report = metrics['classification_report']
        for class_name in class_names:
            if class_name in report:
                class_metrics = report[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n\n")
    
    print(f"Evaluation report saved to {report_path}")


def main():
    """Main evaluation function."""
    config = load_config()
    
    set_seed(42)
    
    device = get_device()
    
    model_name = config['train']['model']
    checkpoint_dir = Path("checkpoints")
    
    best_model_path = checkpoint_dir / f"{model_name}_best.pth"
    final_model_path = checkpoint_dir / f"{model_name}_final.pth"
    
    if best_model_path.exists():
        model_path = best_model_path
        print(f"Loading best model: {model_path}")
    elif final_model_path.exists():
        model_path = final_model_path
        print(f"Loading final model: {model_path}")
    else:
        print("No trained model found. Please run training first.")
        return
    
    model, model_config, class_names = load_trained_model(str(model_path), device)
    print(f"Loaded model with {len(class_names)} classes: {class_names}")
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    
    generate_evaluation_report(test_metrics)
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, class_names)
    
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    save_metrics(val_metrics, "report_assets/validation_metrics.json")
    
    print("\nEvaluation completed! Check report_assets/ for detailed results.")


if __name__ == "__main__":
    main()
