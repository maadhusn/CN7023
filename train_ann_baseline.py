"""ANN baseline training script for MSc coursework."""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

from ann_dataset import create_ann_data_loaders
from src.data import set_seed


class ANNBaseline(nn.Module):
    """Simple MLP for baseline comparison."""
    
    def __init__(self, input_dim=3072, hidden1=128, hidden2=64, num_classes=10, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def test_model(model, test_loader, device):
    """Test the model."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100.0 * correct / total
    return test_acc, all_predictions, all_targets


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('ANN Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('ANN Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('ANN Baseline Confusion Matrix (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    
    plt.xlabel('Plant Disease Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('ANN Baseline Per-Class Accuracy on Test Set')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    avg_acc = np.mean(per_class_acc)
    plt.axhline(y=avg_acc, color='blue', linestyle='--', label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ANN baseline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    dataset_config = config.get('dataset', {})
    ann_config = config.get('ann', {})
    
    dataset_path = dataset_config.get('path', 'C:/PlantVillage')
    downsample_size = ann_config.get('downsample_size', 32)
    hidden1 = ann_config.get('hidden1', 128)
    hidden2 = ann_config.get('hidden2', 64)
    dropout = ann_config.get('dropout', 0.2)
    lr = ann_config.get('lr', 0.001)
    epochs = ann_config.get('epochs', 40)
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs('results', exist_ok=True)
    
    from torchvision import datasets
    temp_dataset = datasets.ImageFolder(root=dataset_path)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    
    print("Creating ANN data loaders...")
    train_loader, val_loader, test_loader = create_ann_data_loaders(
        dataset_path, class_names, batch_size, downsample_size
    )
    
    print(f"ANN Dataset info:")
    print(f"  Input dimension: {downsample_size * downsample_size * 3}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    input_dim = downsample_size * downsample_size * 3
    model = ANNBaseline(input_dim, hidden1, hidden2, num_classes, dropout)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting ANN training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\nTesting ANN model...")
    test_acc, test_predictions, test_targets = test_model(model, test_loader, device)
    print(f"ANN Test Accuracy: {test_acc:.2f}%")
    
    torch.save(model.state_dict(), 'results/ann_baseline_model.pth')
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 'results/ann_curves.png')
    plot_confusion_matrix(test_targets, test_predictions, class_names, 'results/ann_confusion_matrix.png')
    plot_per_class_accuracy(test_targets, test_predictions, class_names, 'results/per_class_accuracy_ann.png')
    
    cm = confusion_matrix(test_targets, test_predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'per_class_acc': per_class_acc.tolist(),
        'class_names': class_names,
        'hyperparameters': {
            'downsample_size': downsample_size,
            'hidden1': hidden1,
            'hidden2': hidden2,
            'dropout': dropout,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size
        }
    }
    
    with open('results/ann_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANN BASELINE SUMMARY FOR REPORT")
    print(f"{'='*60}")
    print(f"ANN Baseline Accuracy: {test_acc:.2f}% (downsampled {downsample_size}x{downsample_size} RGB images)")
    print(f"Architecture: {input_dim} → {hidden1} → {hidden2} → {num_classes} (MLP with ReLU and dropout)")
    print(f"Training completed in {epochs} epochs with Adam optimizer (lr={lr})")
    
    print(f"\nANN baseline training completed!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
