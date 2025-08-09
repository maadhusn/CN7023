"""Evaluate CNN models and generate comprehensive reports."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from aug import apply_test_time_augmentation
from data import create_data_loaders, load_config
from models.cnn import create_model
from utils import (
    calculate_metrics,
    get_device,
    plot_confusion_matrix,
    save_metrics,
    set_seed,
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list,
    use_tta: bool = False,
    config: dict = None
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            if use_tta and config:
                batch_outputs = []
                for i in range(data.size(0)):
                    single_input = data[i:i+1]
                    tta_inputs = apply_test_time_augmentation(single_input[0], config)

                    tta_outputs = []
                    for tta_input in tta_inputs:
                        if isinstance(tta_input, torch.Tensor):
                            tta_input = tta_input.unsqueeze(0).to(device)
                        else:
                            tta_input = torch.tensor(tta_input).unsqueeze(0).to(device)

                        with torch.no_grad():
                            tta_output = model(tta_input)
                            tta_outputs.append(tta_output)

                    # Average TTA predictions
                    avg_output = torch.mean(torch.stack(tta_outputs), dim=0)
                    batch_outputs.append(avg_output)

                output = torch.cat(batch_outputs, dim=0)
            else:
                output = model(data)

            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    metrics = calculate_metrics(all_targets, all_predictions, class_names)

    metrics['probabilities'] = np.array(all_probabilities).tolist()
    metrics['predictions'] = all_predictions
    metrics['targets'] = all_targets

    return metrics


def load_trained_model(model_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint.get('config', load_config())
    classes = checkpoint.get('classes', ['class_0', 'class_1', 'class_2', 'class_3'])

    model_name = config['train']['model']
    num_classes = len(classes)
    model = create_model(model_name, num_classes, pretrained=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, config, classes


def generate_evaluation_report(metrics: dict, save_dir: str = "report_assets"):
    """Generate comprehensive evaluation report."""
    os.makedirs(save_dir, exist_ok=True)

    save_metrics(metrics, os.path.join(save_dir, "evaluation_metrics.json"))

    cm = np.array(metrics['confusion_matrix'])
    class_names = list(metrics['classification_report'].keys())[:-3]  # Remove avg keys

    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png")
    )

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Plant Disease Classification - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")

        f.write("Per-Class Performance:\n")
        f.write("-" * 30 + "\n")

        report = metrics['classification_report']
        for class_name in class_names:
            if class_name in report:
                class_metrics = report[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n\n")

    print(f"Classification report saved to {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate CNN model')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--gradcam', type=int, default=0, help='Generate N GradCAM visualizations')
    args = parser.parse_args()

    config = load_config()

    if args.tta:
        config['eval']['tta'] = True

    set_seed(42)

    device = get_device()

    model_name = config['train']['model']
    checkpoint_dir = Path("checkpoints")

    state_dict_path = checkpoint_dir / f"small_{model_name}_state.pt"
    if state_dict_path.exists():
        print(f"Loading state dict: {state_dict_path}")
        num_classes = len(config.get('classes', ['class_0', 'class_1', 'class_2', 'class_3']))
        model = create_model(model_name, num_classes, pretrained=False)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model = model.to(device)
        class_names = config.get('classes', [f'class_{i}' for i in range(num_classes)])
    else:
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

    use_tta = config['eval'].get('tta', False)
    if use_tta:
        print("Using test-time augmentation")

    print("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, device, class_names, use_tta, config)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")

    generate_evaluation_report(test_metrics)

    metrics_path = "report_assets/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        metrics['test_accuracy'] = test_metrics['accuracy']
        metrics['test_macro_f1'] = test_metrics['macro_f1']
        metrics['test_weighted_f1'] = test_metrics['weighted_f1']
        metrics['test_classification_report'] = test_metrics['classification_report']

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Updated {metrics_path} with test results")

    if args.gradcam > 0:
        from gradcam import generate_individual_gradcam_files
        print(f"Generating {args.gradcam} GradCAM visualizations...")
        generate_individual_gradcam_files(
            model, test_loader, class_names, device, model_name, args.gradcam
        )

    print("\nEvaluation completed! Check report_assets/ for detailed results.")


if __name__ == "__main__":
    main()
