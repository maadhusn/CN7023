"""Data loading and preprocessing utilities with path-preserving policies."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        samples = []
        classes_set = set()
        
        with open(manifest_path, 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                
                class_name = Path(rel_path).parent.name
                classes_set.add(class_name)
                
                full_path = self.root / self.variant / rel_path
                samples.append((str(full_path), class_name))
        
        classes = sorted(list(classes_set))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = [(path, class_to_idx[cls]) for path, cls in samples]
        
        if self.subset_per_class:
            samples = self._apply_subset_sampling(samples, classes)
        
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
        for class_idx, class_samples_list in class_samples.items():
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
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _create_synthetic_image(self) -> Image.Image:
        """Create a synthetic RGB image for testing."""
        size = (224, 224)
        array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(array)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
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
