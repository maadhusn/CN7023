"""Data loading with deterministic manifest-based splits for MSc coursework."""

import os
import argparse
import random
import yaml
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

DATASET_PATH = "C:/PlantVillage"
SPLITS_DIR = "C:/PlantVillage/splits"

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


def create_stratified_splits(dataset_path, splits_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, subset_per_class=None, seed=42):
    """Create stratified train/val/test splits and save to manifest files."""
    set_seed(seed)
    
    os.makedirs(splits_dir, exist_ok=True)
    
    class_samples = defaultdict(list)
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = f"{class_name}/{filename}"
                class_samples[class_name].append(rel_path)
    
    if subset_per_class:
        for class_name in class_samples:
            if len(class_samples[class_name]) > subset_per_class:
                class_samples[class_name] = random.sample(class_samples[class_name], subset_per_class)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for class_name, samples in class_samples.items():
        random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    
    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for sample in test_samples:
            f.write(f"{sample}\n")
    
    print(f"Created stratified splits:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    print(f"  Classes: {len(class_samples)}")
    
    return sorted(class_samples.keys())


class ManifestDataset(Dataset):
    """Dataset that loads images from manifest files."""
    
    def __init__(self, manifest_path, dataset_path, class_names, transform=None):
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        
        self.samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                rel_path = line.strip()
                if rel_path:
                    class_name = rel_path.split('/')[0]
                    if class_name in self.class_to_idx:
                        full_path = self.dataset_path / rel_path
                        self.samples.append((str(full_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def create_data_loaders(batch_size=32, image_size=224, config_path=None):
    """Create train/val/test data loaders using manifest files."""
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    dataset_path = config.get('dataset', {}).get('path', DATASET_PATH)
    splits_dir = os.path.join(dataset_path, 'splits')
    
    train_manifest = os.path.join(splits_dir, 'train.txt')
    val_manifest = os.path.join(splits_dir, 'val.txt')
    test_manifest = os.path.join(splits_dir, 'test.txt')
    
    if not all(os.path.exists(f) for f in [train_manifest, val_manifest, test_manifest]):
        print("Manifest files not found. Creating splits...")
        class_names = create_stratified_splits(
            dataset_path, splits_dir,
            subset_per_class=config.get('dataset', {}).get('subset_per_class')
        )
    else:
        temp_dataset = datasets.ImageFolder(root=dataset_path)
        class_names = temp_dataset.classes
    
    train_transform, val_test_transform = get_transforms(image_size)
    
    train_dataset = ManifestDataset(train_manifest, dataset_path, class_names, train_transform)
    val_dataset = ManifestDataset(val_manifest, dataset_path, class_names, val_test_transform)
    test_dataset = ManifestDataset(test_manifest, dataset_path, class_names, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, class_names


def main():
    """CLI for regenerating splits."""
    parser = argparse.ArgumentParser(description='Generate deterministic dataset splits')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--regen-splits', action='store_true', help='Regenerate splits')
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    dataset_path = dataset_config.get('path', DATASET_PATH)
    splits_dir = os.path.join(dataset_path, 'splits')
    
    if args.regen_splits:
        print(f"Regenerating splits for dataset at: {dataset_path}")
        class_names = create_stratified_splits(
            dataset_path, splits_dir,
            train_ratio=dataset_config.get('train_split', 0.7),
            val_ratio=dataset_config.get('val_split', 0.15),
            test_ratio=dataset_config.get('test_split', 0.15),
            subset_per_class=dataset_config.get('subset_per_class'),
            seed=config.get('seed', 42)
        )
        print(f"Splits saved to: {splits_dir}")
    else:
        print("Use --regen-splits to regenerate dataset splits")


if __name__ == "__main__":
    main()
