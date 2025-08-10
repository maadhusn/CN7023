"""Training script for MSc coursework using SafeImageFolder data loading."""

import os
import argparse
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from pathlib import Path
import json
from src.custom_imagefolder import SafeImageFolder as ImageFolder

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


def create_resnet50_model(num_classes):
    """Create ResNet50 model with custom classifier head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
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
    """Test the model on test set."""
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ResNet50 model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
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
    
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    
    epochs = training_config.get('epochs', 25)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    image_size = dataset_config.get('image_size', 224)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs('results', exist_ok=True)
    
    print("Creating data loaders from manifests...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        batch_size=batch_size, image_size=image_size, config_path=args.config
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print("Creating ResNet50 model...")
    model = create_resnet50_model(num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training for {epochs} epochs...")
    
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
    
    print("\nTesting model...")
    test_acc, test_predictions, test_targets = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    torch.save(model.state_dict(), 'results/resnet50_model.pth')
    print("Model saved to results/resnet50_model.pth")
    
    def _to_py(x):
        try:
            return x.item()
        except Exception:
            if hasattr(x, 'dtype'):  # numpy types
                return x.item() if hasattr(x, 'item') else float(x)
            return float(x) if isinstance(x, (int, float)) else x
    
    results = {
        'train_losses': [_to_py(x) for x in train_losses],
        'val_losses': [_to_py(x) for x in val_losses],
        'train_accs': [_to_py(x) for x in train_accs],
        'val_accs': [_to_py(x) for x in val_accs],
        'test_acc': _to_py(test_acc),
        'test_predictions': [_to_py(x) for x in test_predictions],
        'test_targets': [_to_py(x) for x in test_targets],
        'class_names': class_names,
        'num_classes': num_classes,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'image_size': image_size
        }
    }
    
    with open('results/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("Training results saved to results/training_results.json")
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
