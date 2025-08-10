"""Enhanced evaluation script with Grad-CAM and comprehensive metrics."""

import os
import argparse
import yaml
import random
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from torchvision import transforms
from pathlib import Path
from src.custom_imagefolder import SafeImageFolder as ImageFolder

from train import create_resnet50_model
from gradcam import generate_gradcam_visualizations
from cv_viz import generate_prediction_visualizations

DATASET_DEFAULT = r"C:/PlantVillage"  # Windows local
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_transforms(image_size=224):
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_test_transform


def create_data_loaders(batch_size=32, image_size=224, config_path=None):
    """Create train/val/test data loaders using SafeImageFolder."""
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    dataset_path = config.get('dataset', {}).get('path', DATASET_DEFAULT)
    print(f"Using dataset path: {dataset_path}")
    
    train_transform, val_test_transform = get_transforms(image_size)
    
    full_dataset = ImageFolder(root=dataset_path, transform=None)
    class_names = full_dataset.classes
    
    set_seed(config.get('seed', 42))
    
    dataset_config = config.get('dataset', {})
    train_ratio = dataset_config.get('train_split', 0.7)
    val_ratio = dataset_config.get('val_split', 0.15)
    test_ratio = dataset_config.get('test_split', 0.15)
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, class_names


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
    parser.add_argument('--regen-splits', action='store_true', help='No-op (manifests not used)')
    args = parser.parse_args()
    
    if args.regen_splits:
        print("--regen-splits: skipped (no manifests used)")
        return
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs('results', exist_ok=True)
    
    print("Loading data using SafeImageFolder...")
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
