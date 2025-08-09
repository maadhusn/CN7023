"""Synthetic demo for end-to-end testing without real datasets."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import load_config
from models.cnn import create_model
from utils import get_device, set_seed


def create_synthetic_dataset(
    num_classes: int = 6,
    samples_per_class: int = 20,
    image_size: int = 192,
    save_dir: str = None
) -> str:
    """Create a synthetic dataset for testing."""
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="synthetic_plantvillage_")

    save_dir = Path(save_dir)

    variant_dir = save_dir / "color"
    splits_dir = save_dir / "splits" / "color"

    variant_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["healthy", "bacterial_spot", "early_blight", "late_blight", "mosaic_virus", "target_spot"][:num_classes]

    all_samples = []

    print(f"Creating synthetic dataset in {save_dir}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = variant_dir / class_name
        class_dir.mkdir(exist_ok=True)

        for i in range(samples_per_class):
            image = create_synthetic_plant_image(class_idx, image_size)

            image_name = f"img_{i:03d}.jpg"
            image_path = class_dir / image_name
            image.save(image_path)

            rel_path = f"{class_name}/{image_name}"
            all_samples.append((rel_path, class_idx))

    np.random.shuffle(all_samples)

    train_split = 0.7
    val_split = 0.15

    n_train = int(len(all_samples) * train_split)
    n_val = int(len(all_samples) * val_split)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        manifest_path = splits_dir / f"{split_name}.txt"
        with open(manifest_path, 'w') as f:
            for rel_path, _ in samples:
                f.write(f"{rel_path}\n")

    print(f"Created {len(all_samples)} synthetic images")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    return str(save_dir)


def create_synthetic_plant_image(class_idx: int, size: int = 192) -> Image.Image:
    """Create a synthetic plant image with class-specific characteristics."""
    image = np.random.randint(50, 150, (size, size, 3), dtype=np.uint8)

    image[:, :, 1] = np.clip(image[:, :, 1] + 50, 0, 255)  # More green

    if class_idx == 0:  # Healthy
        image[:, :, 1] = np.clip(image[:, :, 1] + 30, 0, 255)

    elif class_idx == 1:  # Bacterial spot
        num_spots = np.random.randint(5, 15)
        for _ in range(num_spots):
            center_x = np.random.randint(10, size - 10)
            center_y = np.random.randint(10, size - 10)
            radius = np.random.randint(3, 8)

            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

            image[mask] = [30, 20, 10]  # Dark brown spots

    elif class_idx == 2:  # Early blight
        num_patches = np.random.randint(3, 8)
        for _ in range(num_patches):
            center_x = np.random.randint(20, size - 20)
            center_y = np.random.randint(20, size - 20)
            radius = np.random.randint(8, 15)

            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

            image[mask, 0] = np.clip(image[mask, 0] + 80, 0, 255)  # More red
            image[mask, 1] = np.clip(image[mask, 1] + 60, 0, 255)  # More green
            image[mask, 2] = np.clip(image[mask, 2] - 30, 0, 255)  # Less blue

    elif class_idx == 3:  # Late blight
        num_lesions = np.random.randint(2, 6)
        for _ in range(num_lesions):
            center_x = np.random.randint(15, size - 15)
            center_y = np.random.randint(15, size - 15)
            width = np.random.randint(10, 25)
            height = np.random.randint(10, 25)

            x1, x2 = max(0, center_x - width//2), min(size, center_x + width//2)
            y1, y2 = max(0, center_y - height//2), min(size, center_y + height//2)

            image[y1:y2, x1:x2] = image[y1:y2, x1:x2] * 0.3  # Darken the area

    elif class_idx == 4:  # Mosaic virus
        num_patches = np.random.randint(8, 15)
        for _ in range(num_patches):
            center_x = np.random.randint(10, size - 10)
            center_y = np.random.randint(10, size - 10)
            radius = np.random.randint(5, 12)

            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

            image[mask, 1] = np.clip(image[mask, 1] + 40, 0, 255)  # More green
            image[mask, 2] = np.clip(image[mask, 2] + 60, 0, 255)  # More blue

    elif class_idx == 5:  # Target spot
        num_rings = np.random.randint(3, 7)
        for _ in range(num_rings):
            center_x = np.random.randint(20, size - 20)
            center_y = np.random.randint(20, size - 20)
            outer_radius = np.random.randint(8, 15)
            inner_radius = outer_radius - 3

            y, x = np.ogrid[:size, :size]
            outer_mask = (x - center_x)**2 + (y - center_y)**2 <= outer_radius**2
            inner_mask = (x - center_x)**2 + (y - center_y)**2 <= inner_radius**2
            ring_mask = outer_mask & ~inner_mask

            image[ring_mask] = [80, 40, 20]  # Brown ring pattern

    noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(image)


def run_synthetic_demo():
    """Run complete synthetic demo pipeline."""
    print("PlantVillage Synthetic Demo")
    print("=" * 40)

    set_seed(42)

    config = load_config()

    print("\n1. Creating synthetic dataset...")
    dataset_path = create_synthetic_dataset(
        num_classes=4,
        samples_per_class=config['data'].get('subset_per_class', 20),
        image_size=config['data']['image_size']
    )

    config['data']['root'] = dataset_path

    device = get_device()

    print("\n2. Creating and testing model...")
    model = create_model(config['train']['model'], num_classes=4, pretrained=False)
    model = model.to(device)

    dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size'])
    dummy_input = dummy_input.to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        predictions = torch.softmax(output, dim=1)

    print(f"Model output shape: {output.shape}")
    print(f"Sample predictions: {predictions[0].cpu().numpy()}")

    print("\n3. Testing data loading...")
    from data import PlantVillageDataset, get_transforms

    transform = get_transforms(config, is_training=False)

    dataset = PlantVillageDataset(
        root=dataset_path,
        variant="color",
        split="train",
        transform=transform,
        subset_per_class=10  # Small subset for demo
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    for i in range(min(3, len(dataset))):
        image, label = dataset[i]
        print(f"Sample {i}: shape={image.shape}, label={label} ({dataset.classes[label]})")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )

    batch_count = 0
    for batch_images, batch_labels in data_loader:
        batch_count += 1
        print(f"Batch {batch_count}: images={batch_images.shape}, labels={batch_labels.shape}")
        if batch_count >= 2:  # Only test first 2 batches
            break

    print("\n4. Testing model inference...")
    model.eval()
    with torch.no_grad():
        batch_images = batch_images.to(device)
        outputs = model(batch_images)
        predictions = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)

    print(f"Batch predictions: {predicted_classes.cpu().numpy()}")
    print(f"Prediction confidences: {torch.max(predictions, dim=1)[0].cpu().numpy()}")

    print("\n5. Creating output artifacts...")
    os.makedirs("report_assets", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'classes': dataset.classes,
        'demo_run': True
    }

    torch.save(checkpoint, "checkpoints/synthetic_demo_model.pth")

    report_path = "report_assets/synthetic_demo_report.txt"
    with open(report_path, 'w') as f:
        f.write("PlantVillage Synthetic Demo Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Number of classes: {len(dataset.classes)}\n")
        f.write(f"Classes: {', '.join(dataset.classes)}\n")
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(f"Model: {config['train']['model']}\n")
        f.write(f"Image size: {config['data']['image_size']}\n")
        f.write(f"Device: {device}\n")
        f.write("\nDemo completed successfully!\n")

    print("\n6. Demo completed successfully!")
    print(f"   - Synthetic dataset created at: {dataset_path}")
    print("   - Model checkpoint saved to: checkpoints/synthetic_demo_model.pth")
    print(f"   - Report saved to: {report_path}")
    print(f"   - Classes: {dataset.classes}")

    import shutil
    if dataset_path.startswith("/tmp/"):
        print(f"\nCleaning up temporary dataset: {dataset_path}")
        shutil.rmtree(dataset_path)

    return True


if __name__ == "__main__":
    try:
        success = run_synthetic_demo()
        if success:
            print("\n✅ Synthetic demo completed successfully!")
            exit(0)
        else:
            print("\n❌ Synthetic demo failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Synthetic demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
