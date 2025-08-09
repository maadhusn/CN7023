"""Data loading and preprocessing utilities with path-preserving policies."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlantVillageDataset(Dataset):
    """Dataset class following path-preserving policies."""

    def __init__(
        self,
        root: str,
        variant: str = "color",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        use_masks: bool = False,
        masks_subdir: str = "masks",
        splits_dir: str = "splits",
        subset_per_class: Optional[int] = None,
    ):
        self.root = Path(root) if root else None
        self.variant = variant
        self.split = split
        self.transform = transform
        self.use_masks = use_masks
        self.masks_subdir = masks_subdir
        self.splits_dir = splits_dir
        self.subset_per_class = subset_per_class

        if self.root is None or not self.root.exists():
            self.samples, self.classes = self._create_synthetic_data()
        else:
            self.samples, self.classes = self._load_from_manifests()

    def _create_synthetic_data(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Create synthetic data for testing without real dataset."""
        classes = ["healthy", "diseased_a", "diseased_b", "diseased_c"]
        samples = []

        samples_per_class = self.subset_per_class or 20
        for class_idx, class_name in enumerate(classes):
            for i in range(samples_per_class):
                synthetic_path = f"synthetic/{self.variant}/{class_name}/img_{i:03d}.jpg"
                samples.append((synthetic_path, class_idx))

        return samples, classes

    def _load_from_manifests(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load data using manifest files (path-preserving policy)."""
        manifest_path = self.root / self.splits_dir / self.variant / f"{self.split}.txt"

        if not manifest_path.exists():
            classes = discover_classes(self.root, self.variant)
            if classes:
                create_manifests(
                    self.root,
                    self.variant,
                    self.splits_dir,
                    subset_per_class=self.subset_per_class
                )
                save_label_mapping(classes)

                if self.split == 'train':
                    create_eda_visualizations(self.root, self.variant)
            else:
                raise FileNotFoundError(f"Manifest not found and no classes discovered: {manifest_path}")

        samples = []
        classes_set = set()

        with open(manifest_path) as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue

                class_name = Path(rel_path).parts[0]
                classes_set.add(class_name)

                full_path = self.root / self.variant / rel_path
                samples.append((str(full_path), class_name))

        classes = sorted(classes_set)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        samples = [(path, class_to_idx[cls]) for path, cls in samples]

        return samples, classes

    def _apply_subset_sampling(
        self, samples: List[Tuple[str, int]], classes: List[str]
    ) -> List[Tuple[str, int]]:
        """Apply per-class subset sampling."""
        class_samples = {}
        for path, class_idx in samples:
            if class_idx not in class_samples:
                class_samples[class_idx] = []
            class_samples[class_idx].append((path, class_idx))

        subset_samples = []
        for _class_idx, class_samples_list in class_samples.items():
            if len(class_samples_list) > self.subset_per_class:
                sampled = random.sample(class_samples_list, self.subset_per_class)
            else:
                sampled = class_samples_list
            subset_samples.extend(sampled)

        return subset_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        if self.root is None or not Path(path).exists():
            image = self._create_synthetic_image()
        else:
            image = Image.open(path).convert('RGB')

            if self.use_masks:
                mask_path = self._get_mask_path(path)
                if mask_path and mask_path.exists():
                    try:
                        mask = Image.open(mask_path).convert('L')
                        mask_array = np.array(mask) > 128
                        image_array = np.array(image)
                        image_array[~mask_array] = 0
                        image = Image.fromarray(image_array)
                    except Exception:
                        pass

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_mask_path(self, image_path: str) -> Optional[Path]:
        """Get corresponding mask path with .png extension."""
        if not self.root:
            return None

        image_path = Path(image_path)
        rel_path = image_path.relative_to(self.root / self.variant)

        mask_rel_path = rel_path.with_suffix('.png')
        mask_path = self.root / self.masks_subdir / self.variant / mask_rel_path

        return mask_path

    def _create_synthetic_image(self) -> Image.Image:
        """Create a synthetic RGB image for testing."""
        size = (224, 224)
        array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(array)


def discover_classes(root: Path, variant: str) -> List[str]:
    """Discover classes as sorted subfolders under <root>/<variant>/."""
    variant_path = root / variant
    if not variant_path.exists():
        return []

    classes = []
    for item in variant_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            classes.append(item.name)

    return sorted(classes)


def save_label_mapping(classes: List[str], save_dir: str = "report_assets"):
    """Persist class mapping to report_assets/labels.json."""
    os.makedirs(save_dir, exist_ok=True)

    idx_to_class = dict(enumerate(classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    label_mapping = {
        'idx_to_class': idx_to_class,
        'class_to_idx': class_to_idx,
        'num_classes': len(classes)
    }

    labels_path = os.path.join(save_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print(f"Label mapping saved to {labels_path}")
    return label_mapping


def create_manifests(
    root: Path,
    variant: str,
    splits_dir: str = "splits",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    subset_per_class: Optional[int] = None,
    regen_splits: bool = False
):
    """Create or reuse manifests with relative paths."""
    splits_path = root / splits_dir / variant
    splits_path.mkdir(parents=True, exist_ok=True)

    train_manifest = splits_path / "train.txt"
    val_manifest = splits_path / "val.txt"
    test_manifest = splits_path / "test.txt"

    if not regen_splits and all(m.exists() for m in [train_manifest, val_manifest, test_manifest]):
        print(f"Using existing manifests in {splits_path}")
        return

    print(f"Creating manifests in {splits_path}")

    classes = discover_classes(root, variant)
    if not classes:
        raise ValueError(f"No classes found in {root / variant}")

    all_samples = []
    variant_path = root / variant

    for class_name in classes:
        class_path = variant_path / class_name
        if not class_path.exists():
            continue

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_path.glob(ext))
            image_files.extend(class_path.glob(ext.upper()))

        class_samples = []
        for img_path in image_files:
            rel_path = f"{class_name}/{img_path.name}"
            class_samples.append(rel_path)

        if subset_per_class and len(class_samples) > subset_per_class:
            class_samples = random.sample(class_samples, subset_per_class)

        all_samples.extend(class_samples)

    random.shuffle(all_samples)

    n_train = int(len(all_samples) * train_split)
    n_val = int(len(all_samples) * val_split)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    for manifest_path, samples in [
        (train_manifest, train_samples),
        (val_manifest, val_samples),
        (test_manifest, test_samples)
    ]:
        with open(manifest_path, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")

    print(f"Created manifests: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")


def create_eda_visualizations(root: Path, variant: str, save_dir: str = "report_assets"):
    """Create EDA visualizations on first run with real data."""
    os.makedirs(save_dir, exist_ok=True)

    classes = discover_classes(root, variant)
    if not classes:
        print("No classes found for EDA")
        return

    variant_path = root / variant
    class_counts = {}
    sample_images = {}

    for class_name in classes:
        class_path = variant_path / class_name
        if not class_path.exists():
            continue

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_path.glob(ext))
            image_files.extend(class_path.glob(ext.upper()))

        class_counts[class_name] = len(image_files)

        if image_files:
            sample_img = random.choice(image_files)
            sample_images[class_name] = sample_img

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_counts.keys(), class_counts.values())
    ax.set_title('Class Distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=45)

    for bar, count in zip(bars, class_counts.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(class_counts.values()),
                f'{count}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_counts.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if sample_images:
        n_classes = len(sample_images)
        cols = min(4, n_classes)
        rows = (n_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (class_name, img_path) in enumerate(sample_images.items()):
            if idx >= len(axes):
                break

            try:
                img = Image.open(img_path).convert('RGB')
                axes[idx].imshow(img)
                axes[idx].set_title(f'{class_name}\n({class_counts[class_name]} samples)')
                axes[idx].axis('off')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                axes[idx].text(0.5, 0.5, f'Error loading\n{class_name}',
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')

        for idx in range(len(sample_images), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'samples_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"EDA visualizations saved to {save_dir}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_transforms(config: Dict, is_training: bool = True) -> transforms.Compose:
    """Get image transforms based on configuration."""
    transform_list = []

    image_size = config['data']['image_size']
    transform_list.append(transforms.Resize((image_size, image_size)))

    if is_training:
        aug_config = config['aug']

        if aug_config.get('flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(0.5))

        if aug_config.get('rotation_deg', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(aug_config['rotation_deg'])
            )

        if aug_config.get('color_jitter', 0) > 0:
            jitter = aug_config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=jitter/2
                )
            )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transform_list)


def create_data_loaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test data loaders."""
    data_config = config['data']
    train_config = config['train']

    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)

    train_dataset = PlantVillageDataset(
        root=data_config['root'],
        variant=data_config['variant'],
        split='train',
        transform=train_transform,
        use_masks=data_config['use_masks'],
        masks_subdir=data_config['masks_subdir'],
        splits_dir=data_config['splits_dir'],
        subset_per_class=data_config['subset_per_class'],
    )

    val_dataset = PlantVillageDataset(
        root=data_config['root'],
        variant=data_config['variant'],
        split='val',
        transform=val_transform,
        use_masks=data_config['use_masks'],
        masks_subdir=data_config['masks_subdir'],
        splits_dir=data_config['splits_dir'],
        subset_per_class=data_config['subset_per_class'],
    )

    test_dataset = PlantVillageDataset(
        root=data_config['root'],
        variant=data_config['variant'],
        split='test',
        transform=val_transform,
        use_masks=data_config['use_masks'],
        masks_subdir=data_config['masks_subdir'],
        splits_dir=data_config['splits_dir'],
        subset_per_class=data_config['subset_per_class'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description='Data processing with path-preserving policies')
    parser.add_argument('--regen-splits', action='store_true',
                       help='Regenerate train/val/test split manifests')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()
    config = load_config(args.config)

    if config['data']['root']:
        root = Path(config['data']['root'])
        variant = config['data']['variant']

        if root.exists():
            classes = discover_classes(root, variant)
            if classes:
                create_manifests(
                    root,
                    variant,
                    config['data']['splits_dir'],
                    config['data'].get('train_split', 0.70),
                    config['data'].get('val_split', 0.15),
                    config['data'].get('test_split', 0.15),
                    config['data']['subset_per_class'],
                    regen_splits=args.regen_splits
                )
                save_label_mapping(classes)
                create_eda_visualizations(root, variant)
                print("Data processing completed!")
            else:
                print("No classes found in dataset")
        else:
            print(f"Dataset root does not exist: {root}")
    else:
        print("No dataset root specified in config")


if __name__ == "__main__":
    main()
