"""ANN dataset for downsampled images from manifest files."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ANNDataset(Dataset):
    """Dataset for ANN training with downsampled images."""
    
    def __init__(self, manifest_path, dataset_path, class_names, downsample_size=32):
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.downsample_size = downsample_size
        
        self.transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.ToTensor(),
        ])
        
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
        
        image = self.transform(image)
        image = image.view(-1)  # Flatten to 1D tensor (3 * 32 * 32 = 3072)
        
        return image, target


def create_ann_data_loaders(dataset_path, class_names, batch_size=32, downsample_size=32):
    """Create ANN data loaders from manifest files."""
    splits_dir = os.path.join(dataset_path, 'splits')
    
    train_manifest = os.path.join(splits_dir, 'train.txt')
    val_manifest = os.path.join(splits_dir, 'val.txt')
    test_manifest = os.path.join(splits_dir, 'test.txt')
    
    train_dataset = ANNDataset(train_manifest, dataset_path, class_names, downsample_size)
    val_dataset = ANNDataset(val_manifest, dataset_path, class_names, downsample_size)
    test_dataset = ANNDataset(test_manifest, dataset_path, class_names, downsample_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
