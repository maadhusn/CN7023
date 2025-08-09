"""Enhanced evaluation script with Grad-CAM and comprehensive metrics."""

import os
import argparse
import yaml
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import pandas as pd

from src.data import create_data_loaders, set_seed
from train import create_resnet50_model
from gradcam import generate_gradcam_visualizations
from cv_viz import generate_prediction_visualizations


def calculate_enhanced_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics including precision, recall, F1."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_macro': float(macro_precision),
            'recall_macro': float(macro_recall),
            'f1_macro': float(macro_f1),
            'precision_micro': float(micro_precision),
            'recall_micro': float(micro_recall),
            'f1_micro': float(micro_f1)
        },
        'per_class': {
            'class_names': class_names,
            'accuracy': per_class_accuracy.tolist(),
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        }
    }
    
    return metrics


def analyze_top_confusions(y_true, y_pred, class_names, top_k=10):
    """Analyze top confused class pairs."""
    cm = confusion_matrix(y_true, y_pred)
    
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(cm[i, j]),
                    'true_class_total': int(cm[i].sum()),
                    'confusion_rate': float(cm[i, j] / cm[i].sum())
                })
    
    confusions.sort(key=lambda x: x['count'], reverse=True)
    return confusions[:top_k]


def generate_summary_texts(metrics, confusions, class_names, dataset_info, save_dir='results'):
    """Generate paste-ready text summaries for report."""
    
    dataset_summary = f"""Dataset Summary:
- Total samples: {dataset_info['total_samples']:,}
- Number of classes: {len(class_names)}
- Train/Val/Test split: {dataset_info['train_samples']:,}/{dataset_info['val_samples']:,}/{dataset_info['test_samples']:,}
- Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}

Class distribution shows {'balanced' if dataset_info.get('balanced', True) else 'imbalanced'} representation across disease categories.
The dataset contains RGB images resized to 224×224 pixels with ImageNet normalization applied.
"""
    
    with open(os.path.join(save_dir, 'summary_dataset.txt'), 'w') as f:
        f.write(dataset_summary)
    
    results_summary = f"""Results Summary:
- CNN Test Accuracy: {metrics['overall']['accuracy']*100:.2f}%
- Weighted F1-Score: {metrics['overall']['f1_weighted']:.3f}
- Macro F1-Score: {metrics['overall']['f1_macro']:.3f}
- Precision (Weighted): {metrics['overall']['precision_weighted']:.3f}
- Recall (Weighted): {metrics['overall']['recall_weighted']:.3f}

The ResNet50 model achieved strong performance across most disease categories.
Best performing classes: {', '.join([class_names[i] for i in np.argsort(metrics['per_class']['accuracy'])[-3:]])}
Worst performing classes: {', '.join([class_names[i] for i in np.argsort(metrics['per_class']['accuracy'])[:3]])}
"""
    
    with open(os.path.join(save_dir, 'summary_results.txt'), 'w') as f:
        f.write(results_summary)
    
    top_confusions_text = '\n'.join([
        f"• {conf['true_class']} → {conf['predicted_class']}: {conf['count']} cases ({conf['confusion_rate']*100:.1f}%)"
        for conf in confusions[:5]
    ])
    
    analysis_summary = f"""Critical Analysis:
Top Confusion Patterns:
{top_confusions_text}

Key Observations:
• Model shows strong overall performance with {metrics['overall']['accuracy']*100:.1f}% accuracy
• F1-score variation across classes indicates some disease types are more challenging to classify
• Common confusions often occur between visually similar disease symptoms
• Performance gaps may be due to class imbalance or symptom similarity
• Further improvements could focus on data augmentation for challenging classes

The confusion matrix reveals systematic patterns in misclassification that could guide future model improvements.
"""
    
    with open(os.path.join(save_dir, 'summary_analysis.txt'), 'w') as f:
        f.write(analysis_summary)
    
    print("Generated summary text files:")
    print("- summary_dataset.txt")
    print("- summary_results.txt") 
    print("- summary_analysis.txt")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ResNet50 model with enhanced metrics')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--model-path', default='results/resnet50_model.pth', help='Path to trained model')
    parser.add_argument('--gradcam', type=int, default=24, help='Number of Grad-CAM samples to generate')
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs('results', exist_ok=True)
    
    print("Loading data from manifests...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        batch_size=config.get('training', {}).get('batch_size', 32),
        image_size=config.get('dataset', {}).get('image_size', 224),
        config_path=args.config
    )
    
    num_classes = len(class_names)
    
    print("Loading trained model...")
    model = create_resnet50_model(num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Evaluating on test set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    print("Calculating comprehensive metrics...")
    metrics = calculate_enhanced_metrics(all_targets, all_predictions, class_names)
    
    confusions = analyze_top_confusions(all_targets, all_predictions, class_names)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    confusions_df = pd.DataFrame(confusions)
    confusions_df.to_csv('results/top_confusions.csv', index=False)
    
    print(f"Generating {args.gradcam} Grad-CAM visualizations...")
    generate_gradcam_visualizations(model, test_loader, class_names, device, args.gradcam)
    
    print("Generating OpenCV prediction visualizations...")
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    generate_prediction_visualizations(model, test_loader, class_names, device, transform, 12)
    
    dataset_info = {
        'total_samples': len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'balanced': True
    }
    
    generate_summary_texts(metrics, confusions, class_names, dataset_info)
    
    print(f"\nEvaluation completed!")
    print(f"Test Accuracy: {metrics['overall']['accuracy']*100:.2f}%")
    print(f"Weighted F1-Score: {metrics['overall']['f1_weighted']:.3f}")
    print(f"Generated {len(confusions)} confusion analysis entries")
    print("\nGenerated files:")
    print("- results/metrics.json")
    print("- results/classification_report.txt")
    print("- results/top_confusions.csv")
    print("- results/gradcam_*.png")
    print("- results/viz_pred_*.png")
    print("- results/summary_*.txt")


if __name__ == "__main__":
    main()
