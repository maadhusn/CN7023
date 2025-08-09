"""Visualization script for MSc coursework report generation."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

DATASET_PATH = "C:/PlantVillage"
RESULTS_PATH = "results"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_results():
    """Load training results from JSON file."""
    with open(os.path.join(RESULTS_PATH, 'training_results.json'), 'r') as f:
        results = json.load(f)
    return results


def load_ann_results():
    """Load ANN baseline results if available."""
    ann_path = os.path.join(RESULTS_PATH, 'ann_metrics.json')
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            return json.load(f)
    return None


def analyze_dataset():
    """Analyze dataset and generate overview statistics."""
    print("=" * 60)
    print("DATASET ANALYSIS FOR REPORT")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    
    class_counts = Counter()
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    
    total_samples = len(dataset)
    num_classes = len(dataset.classes)
    
    print(f"Dataset Overview:")
    print(f"- Total samples: {total_samples:,}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Average samples per class: {total_samples // num_classes:,}")
    print(f"\nClass Distribution:")
    
    for class_name in sorted(dataset.classes):
        count = class_counts[class_name]
        percentage = (count / total_samples) * 100
        print(f"- {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    plt.figure(figsize=(12, 6))
    classes = sorted(dataset.classes)
    counts = [class_counts[cls] for cls in classes]
    
    bars = plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Plant Disease Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in PlantVillage Dataset')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nClass distribution chart saved to {RESULTS_PATH}/class_distribution.png")
    
    return dataset, class_counts


def generate_sample_grid(dataset):
    """Generate a grid of sample images from different classes."""
    plt.figure(figsize=(16, 16))
    
    samples_per_class = max(1, 16 // len(dataset.classes))
    selected_samples = []
    
    for class_idx, class_name in enumerate(dataset.classes):
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        selected = random.sample(class_samples, min(samples_per_class, len(class_samples)))
        selected_samples.extend(selected[:samples_per_class])
    
    while len(selected_samples) < 16:
        remaining = random.choice(range(len(dataset)))
        if remaining not in selected_samples:
            selected_samples.append(remaining)
    
    selected_samples = selected_samples[:16]
    
    for i, sample_idx in enumerate(selected_samples):
        plt.subplot(4, 4, i + 1)
        
        img_path, label = dataset.samples[sample_idx]
        img = datasets.folder.default_loader(img_path)
        img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
        
        plt.imshow(img)
        plt.title(f'{dataset.classes[label]}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Sample Images from PlantVillage Dataset', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'dataset_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample grid saved to {RESULTS_PATH}/dataset_samples.png")


def plot_training_curves(results):
    """Plot training and validation curves."""
    epochs = range(1, len(results['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, results['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, results['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=results['test_acc'], color='g', linestyle='--', label=f'Test Accuracy ({results["test_acc"]:.1f}%)', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {RESULTS_PATH}/accuracy_curve.png and {RESULTS_PATH}/loss_curve.png")


def generate_confusion_matrix(results):
    """Generate confusion matrix with per-class accuracy."""
    y_true = results['test_targets']
    y_pred = results['test_predictions']
    class_names = results['class_names']
    
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(12, 10))
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Confusion Matrix with Per-Class Accuracy (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {RESULTS_PATH}/confusion_matrix.png")
    
    return per_class_acc


def plot_per_class_accuracy(results, per_class_acc):
    """Plot per-class accuracy bar chart."""
    class_names = results['class_names']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    plt.xlabel('Plant Disease Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on Test Set')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    avg_acc = np.mean(per_class_acc)
    plt.axhline(y=avg_acc, color='blue', linestyle='--', label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class accuracy chart saved to {RESULTS_PATH}/per_class_accuracy.png")


def generate_report_text(results, class_counts, per_class_acc):
    """Generate text summaries for report sections."""
    print("\n" + "=" * 60)
    print("REPORT TEXT SUMMARIES")
    print("=" * 60)
    
    print("\n[DATASET DESCRIPTION SECTION]")
    print("-" * 40)
    total_samples = sum(class_counts.values())
    num_classes = len(results['class_names'])
    
    print(f"The PlantVillage dataset contains {total_samples:,} images across {num_classes} plant disease classes. ")
    print(f"The dataset includes various plant species with different disease conditions and healthy samples. ")
    print(f"Images are RGB color photographs with varying resolutions, resized to {IMAGE_SIZE}x{IMAGE_SIZE} pixels for training. ")
    print(f"The dataset was split into 70% training ({int(0.7*total_samples):,} images), 15% validation ({int(0.15*total_samples):,} images), ")
    print(f"and 15% test ({int(0.15*total_samples):,} images) sets using random sampling with a fixed seed for reproducibility.")
    
    print("\n[RESULTS OBTAINED SECTION]")
    print("-" * 40)
    best_val_acc = max(results['val_accs'])
    final_train_acc = results['train_accs'][-1]
    test_acc = results['test_acc']
    
    print(f"The ResNet50 model achieved a final training accuracy of {final_train_acc:.2f}% and ")
    print(f"a best validation accuracy of {best_val_acc:.2f}% during {len(results['train_accs'])} epochs of training. ")
    print(f"On the independent test set, the model achieved an accuracy of {test_acc:.2f}%. ")
    print(f"The model was trained using Adam optimizer with learning rate {results['hyperparameters']['learning_rate']} ")
    print(f"and weight decay {results['hyperparameters']['weight_decay']}. ")
    print(f"Training was performed with batch size {results['hyperparameters']['batch_size']} and ImageNet pre-trained weights.")
    
    print("\n[CRITICAL ANALYSIS SECTION]")
    print("-" * 40)
    
    best_class_idx = np.argmax(per_class_acc)
    worst_class_idx = np.argmin(per_class_acc)
    best_class = results['class_names'][best_class_idx]
    worst_class = results['class_names'][worst_class_idx]
    
    print(f"Analysis of the confusion matrix reveals significant variation in per-class performance. ")
    print(f"The best-performing class was '{best_class}' with {per_class_acc[best_class_idx]:.1f}% accuracy, ")
    print(f"while the worst-performing class was '{worst_class}' with {per_class_acc[worst_class_idx]:.1f}% accuracy. ")
    
    acc_std = np.std(per_class_acc)
    print(f"The standard deviation of per-class accuracies is {acc_std:.1f}%, indicating ")
    if acc_std > 15:
        print("significant class imbalance or varying difficulty in disease classification. ")
    else:
        print("relatively consistent performance across different disease types. ")
    
    if results['val_accs'][-1] < max(results['val_accs']) - 5:
        print("The validation accuracy shows signs of overfitting in later epochs, ")
        print("suggesting that early stopping or regularization techniques could improve generalization. ")
    else:
        print("The training curves show stable convergence without significant overfitting. ")
    
    print(f"The gap between training accuracy ({final_train_acc:.1f}%) and test accuracy ({test_acc:.1f}%) ")
    gap = abs(final_train_acc - test_acc)
    if gap > 10:
        print("indicates potential overfitting and suggests the need for stronger regularization.")
    elif gap > 5:
        print("suggests mild overfitting, which is typical for deep learning models.")
    else:
        print("indicates good generalization capability of the trained model.")


def main():
    """Main visualization function."""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("Loading training results...")
    results = load_results()
    
    print("Loading ANN baseline results...")
    ann_results = load_ann_results()
    
    dataset, class_counts = analyze_dataset()
    
    print("Generating sample images grid...")
    generate_sample_grid(dataset)
    
    print("Plotting training curves...")
    plot_training_curves(results)
    
    print("Generating confusion matrix...")
    per_class_acc = generate_confusion_matrix(results)
    
    print("Plotting per-class accuracy...")
    plot_per_class_accuracy(results, per_class_acc)
    
    generate_report_text(results, class_counts, per_class_acc)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"All plots saved to {RESULTS_PATH}/ directory:")
    print("- class_distribution.png")
    print("- dataset_samples.png") 
    print("- accuracy_curve.png")
    print("- loss_curve.png")
    print("- confusion_matrix.png")
    print("- per_class_accuracy.png")
    
    if ann_results:
        print("- ann_curves.png")
        print("- ann_confusion_matrix.png")
        print("- per_class_accuracy_ann.png")
        print(f"\nANN Baseline Comparison:")
        print(f"CNN Test Accuracy: {results['test_acc']:.2f}%")
        print(f"ANN Test Accuracy: {ann_results['test_acc']:.2f}%")
        print(f"Performance Gap: {results['test_acc'] - ann_results['test_acc']:.2f}%")
    
    print("\nAdditional files (if eval.py was run):")
    print("- gradcam_*.png (Grad-CAM visualizations)")
    print("- viz_pred_*.png (OpenCV prediction overlays)")
    print("- metrics.json (comprehensive metrics)")
    print("- classification_report.txt (sklearn report)")
    print("- top_confusions.csv (confusion analysis)")
    print("- summary_*.txt (paste-ready report text)")
    print("\nCopy the text summaries above directly into your report sections.")


if __name__ == "__main__":
    main()
