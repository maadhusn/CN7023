"""Advanced augmentation utilities using Albumentations."""

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_albumentations_transforms(config: dict, is_training: bool = True):
    """Get Albumentations transforms based on configuration."""
    image_size = config['data']['image_size']
    
    if is_training:
        aug_config = config['aug']
        
        transforms_list = [
            A.Resize(image_size, image_size),
        ]
        
        if aug_config.get('flip', False):
            transforms_list.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('rotation_deg', 0) > 0:
            transforms_list.append(
                A.Rotate(
                    limit=aug_config['rotation_deg'],
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )
        
        if aug_config.get('color_jitter', 0) > 0:
            jitter = aug_config['color_jitter']
            transforms_list.extend([
                A.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=jitter/2,
                    p=0.5
                )
            ])
        
        if aug_config.get('blur_p', 0) > 0:
            transforms_list.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=aug_config['blur_p'])
            )
        
        transforms_list.extend([
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
            ], p=0.1),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
            ], p=0.2),
        ])
    
    else:
        transforms_list = [
            A.Resize(image_size, image_size),
        ]
    
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def apply_test_time_augmentation(image: np.ndarray, config: dict) -> list:
    """Apply test-time augmentation for improved inference."""
    if not config['eval'].get('tta', False):
        return [image]
    
    image_size = config['data']['image_size']
    
    tta_transforms = [
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=5, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    augmented_images = []
    for transform in tta_transforms:
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images


def visualize_augmentations(image: np.ndarray, config: dict, num_samples: int = 8):
    """Visualize augmentation effects for debugging."""
    import matplotlib.pyplot as plt
    
    transform = get_albumentations_transforms(config, is_training=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        augmented = transform(image=image)['image']
        
        if hasattr(augmented, 'numpy'):
            augmented = augmented.numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = augmented.transpose(1, 2, 0)
        augmented = augmented * std + mean
        augmented = np.clip(augmented, 0, 1)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmentation {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('report_assets/augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Augmentation samples saved to report_assets/augmentation_samples.png")
