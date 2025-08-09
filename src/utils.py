"""Utility functions for training, evaluation, and visualization."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device
) -> Tuple[int, float, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict:
    """Calculate comprehensive classification metrics."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "report_assets/confusion_matrix.png",
    normalize: bool = True
):
    """Plot and save confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "report_assets/training_history.png"
):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {save_path}")


def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_synthetic_samples(
    num_classes: int = 4,
    samples_per_class: int = 20,
    image_size: int = 224,
    save_dir: str = "synthetic_data"
) -> List[Tuple[str, int]]:
    """Create synthetic image samples for testing."""
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = [f"class_{i}" for i in range(num_classes)]
    samples = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            
            if class_idx == 0:  # Healthy - more green
                image[:, :, 1] = np.clip(image[:, :, 1] + 50, 0, 255)
            elif class_idx == 1:  # Disease A - more brown/yellow
                image[:, :, 0] = np.clip(image[:, :, 0] + 30, 0, 255)
                image[:, :, 1] = np.clip(image[:, :, 1] + 30, 0, 255)
            elif class_idx == 2:  # Disease B - more red spots
                spots = np.random.choice([0, 1], size=(image_size, image_size), p=[0.9, 0.1])
                image[:, :, 0] = np.where(spots, 255, image[:, :, 0])
            
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            image_path = os.path.join(class_dir, f"img_{i:03d}.jpg")
            pil_image.save(image_path)
            
            samples.append((image_path, class_idx))
    
    return samples


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save the best model weights."""
        self.best_weights = model.state_dict().copy()


def log_system_info():
    """Log system information for debugging."""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 30)
