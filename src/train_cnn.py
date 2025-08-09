"""Train CNN models for plant disease classification."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import create_data_loaders, load_config
from models.cnn import create_model, get_model_info
from utils import (
    EarlyStopping,
    get_device,
    log_system_info,
    plot_training_history,
    save_checkpoint,
    set_seed,
)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    config = load_config()
    
    set_seed(42)
    
    log_system_info()
    
    device = get_device()
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_loader.dataset.classes}")
    
    model_name = config['train']['model']
    print(f"Creating model: {model_name}")
    model = create_model(model_name, num_classes, pretrained=True)
    model = model.to(device)
    
    model_info = get_model_info(model)
    print(f"Model parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    scaler = None
    if config['train'].get('mixed_precision', False) and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    
    early_stopping = EarlyStopping(
        patience=config['train']['early_stop_patience'],
        restore_best_weights=True
    )
    
    num_epochs = config['train']['epochs']
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/{model_name}_epoch_{epoch+1}.pth"
        is_best = len(val_losses) == 1 or val_loss < min(val_losses[:-1])
        
        save_checkpoint(
            model, optimizer, epoch+1, val_loss, val_acc,
            checkpoint_path, is_best
        )
        
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    final_model_path = f"checkpoints/{model_name}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'classes': train_loader.dataset.classes,
        'model_info': model_info,
    }, final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
