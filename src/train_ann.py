"""Train ANN models using handcrafted features."""

import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import create_data_loaders, load_config
from features import extract_features_from_loader
from models.ann import MLP, HOGClassifier
from utils import calculate_metrics, get_device, plot_confusion_matrix, save_metrics, set_seed


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


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    num_classes: int,
    device: torch.device
) -> Tuple:
    """Train MLP neural network."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    input_dim = X_train_scaled.shape[1]
    model = MLP(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    num_epochs = 50
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/small_ann_state.pt")
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load("checkpoints/small_ann_state.pt"))
    
    with open("checkpoints/ann_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, train_losses, val_losses, train_accs, val_accs


def main():
    """Main ANN training function."""
    config = load_config()
    
    set_seed(42)
    device = get_device()
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print("Extracting features...")
    X_train, y_train, _ = extract_features_from_loader(train_loader, config, "train")
    X_val, y_val, _ = extract_features_from_loader(val_loader, config, "val")
    X_test, y_test, _ = extract_features_from_loader(test_loader, config, "test")
    
    print(f"Feature extraction completed:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    num_classes = len(np.unique(y_train))
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"\nTraining MLP with {X_train.shape[1]} features for {num_classes} classes...")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("report_assets", exist_ok=True)
    
    model, train_losses, val_losses, train_accs, val_accs = train_mlp(
        X_train, y_train, X_val, y_val, config, num_classes, device
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('ANN Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('ANN Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('report_assets/ann_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("ANN training curves saved to report_assets/ann_curves.png")
    
    with open("checkpoints/ann_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    X_test_scaled = scaler.transform(X_test)
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    test_loader_ann = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader_ann:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    best_val_acc = max(val_accs)
    
    metrics = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'epochs_trained': len(train_losses),
        'feature_dim': X_train.shape[1],
        'num_classes': num_classes,
        'model_architecture': '128->64->num_classes',
        'dropout': 0.2
    }
    
    with open('report_assets/ann_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("ANN metrics saved to report_assets/ann_metrics.json")
    
    print("\nANN training completed!")
    print("Check report_assets/ for detailed results.")


if __name__ == "__main__":
    main()
