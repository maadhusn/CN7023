"""Train CNN models for plant disease classification."""

import json
import os
import warnings

import matplotlib.pyplot as plt
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
    save_checkpoint,
    set_seed,
)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for _batch_idx, (data, target) in enumerate(pbar):
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

        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
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

            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
            )

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

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

    model_name = config["train"]["model"]
    print(f"Creating model: {model_name}")
    model = create_model(model_name, num_classes, pretrained=True)
    model = model.to(device)

    model_info = get_model_info(model)
    print(f"Model parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    num_epochs = config["train"]["epochs"]

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=config["train"]["lr"] * 0.01
    )

    scaler = None
    if config["train"].get("mixed_precision", False) and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")

    early_stopping = EarlyStopping(
        patience=config["train"]["early_stop_patience"], restore_best_weights=True
    )
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

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
            model, optimizer, epoch + 1, val_loss, val_acc, checkpoint_path, is_best
        )

        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    os.makedirs("report_assets", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs_range = range(1, len(train_losses) + 1)

    ax1.plot(epochs_range, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs_range, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs_range, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("report_assets/curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training curves saved to report_assets/curves.png")

    best_val_acc = max(val_accs)
    best_val_loss = min(val_losses)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_train_accuracy": train_accs[-1],
        "final_val_accuracy": val_accs[-1],
        "epochs_trained": len(train_losses),
        "hyperparameters": {
            "model": model_name,
            "learning_rate": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "batch_size": config["train"]["batch_size"],
            "epochs": config["train"]["epochs"],
            "early_stop_patience": config["train"]["early_stop_patience"],
        },
        "class_count": num_classes,
        "classes": train_loader.dataset.classes,
        "model_info": model_info,
    }

    with open("report_assets/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to report_assets/metrics.json")

    state_dict_path = f"checkpoints/small_{model_name}_state.pt"
    torch.save(model.state_dict(), state_dict_path)

    state_dict_size = os.path.getsize(state_dict_path) / (1024 * 1024)  # MB
    if state_dict_size > 80:
        warnings.warn(
            f"State dict checkpoint is {state_dict_size:.1f}MB (>80MB)", stacklevel=2
        )
    elif state_dict_size > 50:
        print(f"Warning: State dict checkpoint is {state_dict_size:.1f}MB (>50MB)")

    print(f"State dict saved to: {state_dict_path} ({state_dict_size:.1f}MB)")

    try:
        model.eval()
        dummy_input = torch.randn(
            1, 3, config["data"]["image_size"], config["data"]["image_size"]
        ).to(device)
        onnx_path = f"checkpoints/small_{model_name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        if onnx_size > 80:
            warnings.warn(f"ONNX model is {onnx_size:.1f}MB (>80MB)", stacklevel=2)
        elif onnx_size > 50:
            print(f"Warning: ONNX model is {onnx_size:.1f}MB (>50MB)")

        print(f"ONNX model saved to: {onnx_path} ({onnx_size:.1f}MB)")
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")

    try:
        model.eval()
        dummy_input = torch.randn(
            1, 3, config["data"]["image_size"], config["data"]["image_size"]
        ).to(device)
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = f"checkpoints/small_{model_name}.torchscript.pt"
        traced_model.save(torchscript_path)

        torchscript_size = os.path.getsize(torchscript_path) / (1024 * 1024)  # MB
        if torchscript_size > 80:
            warnings.warn(
                f"TorchScript model is {torchscript_size:.1f}MB (>80MB)", stacklevel=2
            )
        elif torchscript_size > 50:
            print(f"Warning: TorchScript model is {torchscript_size:.1f}MB (>50MB)")

        print(
            f"TorchScript model saved to: {torchscript_path} ({torchscript_size:.1f}MB)"
        )
    except Exception as e:
        print(f"Failed to export TorchScript model: {e}")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("All artifacts saved to checkpoints/ and report_assets/")


if __name__ == "__main__":
    main()
