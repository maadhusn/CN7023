"""GradCAM visualization for CNN interpretability."""

import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import create_data_loaders, load_config
from eval_cnn import load_trained_model
from utils import get_device, set_seed


class GradCAM:
    """GradCAM implementation for CNN visualization."""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate_cam(
        self, input_tensor: torch.Tensor, class_idx: int = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap."""
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()


def get_target_layer(model: nn.Module, model_name: str) -> str:
    """Get the appropriate target layer for GradCAM."""
    if "efficientnet" in model_name.lower():
        return "backbone.features.7"  # Last conv block
    elif "mobilenet" in model_name.lower():
        return "backbone.features.12"  # Last conv block
    else:
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)

        if conv_layers:
            return conv_layers[-1]
        else:
            raise ValueError("No convolutional layers found in model")


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = image * (1 - alpha) + heatmap_colored * alpha

    return overlayed.astype(np.uint8)


def generate_individual_gradcam_files(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    model_name: str,
    num_samples: int = 4,
    save_dir: str = "report_assets",
):
    """Generate individual GradCAM files with original basenames."""
    os.makedirs(save_dir, exist_ok=True)

    target_layer = get_target_layer(model, model_name)
    print(f"Using target layer: {target_layer}")

    gradcam = GradCAM(model, target_layer)

    samples_collected = 0

    for _batch_idx, (data, targets) in enumerate(data_loader):
        if samples_collected >= num_samples:
            break

        data, targets = data.to(device), targets.to(device)

        for i in range(data.size(0)):
            if samples_collected >= num_samples:
                break

            input_tensor = data[i : i + 1]
            target = targets[i].item()

            input_tensor.requires_grad_(True)
            model.train()  # Set to train mode to enable gradients

            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, predicted_class].item()

            cam = gradcam.generate_cam(input_tensor, predicted_class)

            model.eval()  # Set back to eval mode

            input_image = input_tensor[0].detach().cpu().numpy()
            input_image = input_image.transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_image = input_image * std + mean
            input_image = np.clip(input_image, 0, 1)
            input_image = (input_image * 255).astype(np.uint8)

            overlay = overlay_heatmap(input_image, cam)

            basename = f"sample_{samples_collected:03d}"
            if predicted_class == target:
                filename = f"gradcam_correct_{basename}.png"
            else:
                pred_name = class_names[predicted_class]
                true_name = class_names[target]
                filename = f"gradcam_missed_{basename}__{pred_name}_vs_{true_name}.png"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(input_image)
            axes[0].set_title(f"Original\nTrue: {class_names[target]}")
            axes[0].axis("off")

            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"GradCAM\nPred: {class_names[predicted_class]}")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nConf: {confidence:.3f}")
            axes[2].axis("off")

            plt.tight_layout()
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved GradCAM: {filepath}")
            samples_collected += 1

    print(f"Generated {samples_collected} individual GradCAM files")


def visualize_gradcam_samples(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    model_name: str,
    num_samples: int = 4,
    save_dir: str = "report_assets",
):
    """Generate GradCAM visualizations for sample images."""
    os.makedirs(save_dir, exist_ok=True)

    target_layer = get_target_layer(model, model_name)
    print(f"Using target layer: {target_layer}")

    gradcam = GradCAM(model, target_layer)

    model.eval()
    samples_collected = 0

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for _batch_idx, (data, targets) in enumerate(data_loader):
            if samples_collected >= num_samples:
                break

            data, targets = data.to(device), targets.to(device)

            for i in range(data.size(0)):
                if samples_collected >= num_samples:
                    break

                input_tensor = data[i : i + 1]
                target = targets[i].item()

                with torch.enable_grad():
                    output = model(input_tensor)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = F.softmax(output, dim=1)[0, predicted_class].item()

                cam = gradcam.generate_cam(input_tensor, predicted_class)

                input_image = input_tensor[0].detach().cpu().numpy()
                input_image = input_image.transpose(1, 2, 0)

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_image = input_image * std + mean
                input_image = np.clip(input_image, 0, 1)
                input_image = (input_image * 255).astype(np.uint8)

                overlay = overlay_heatmap(input_image, cam)

                row = samples_collected

                axes[row, 0].imshow(input_image)
                axes[row, 0].set_title(f"Original\nTrue: {class_names[target]}")
                axes[row, 0].axis("off")

                axes[row, 1].imshow(cam, cmap="jet")
                axes[row, 1].set_title(f"GradCAM\nPred: {class_names[predicted_class]}")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(overlay)
                axes[row, 2].set_title(f"Overlay\nConf: {confidence:.3f}")
                axes[row, 2].axis("off")

                samples_collected += 1

    plt.tight_layout()
    gradcam_path = os.path.join(save_dir, "gradcam_visualizations.png")
    plt.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"GradCAM visualizations saved to {gradcam_path}")


def main():
    """Main GradCAM visualization function."""
    config = load_config()

    set_seed(42)

    device = get_device()

    model_name = config["train"]["model"]
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

    num_samples = config["eval"].get("gradcam_samples", 4)
    print(f"Generating GradCAM visualizations for {num_samples} samples...")

    visualize_gradcam_samples(
        model, test_loader, class_names, device, model_name, num_samples
    )

    print("GradCAM visualization completed!")


if __name__ == "__main__":
    main()
