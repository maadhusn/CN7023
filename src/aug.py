"""Augmentation utilities with torchvision transforms."""

import torch
from torchvision import transforms


def get_train_aug(size: int) -> transforms.Compose:
    """Get training augmentations: flip/rotate/jitter/optional blur + normalize (ImageNet)."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_aug(size: int) -> transforms.Compose:
    """Get validation augmentations: resize/center-crop + normalize only."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def apply_test_time_augmentation(image: torch.Tensor, config: dict) -> list:
    """Apply test-time augmentation for improved inference."""
    if not config["eval"].get("tta", False):
        return [image]

    image_size = config["data"]["image_size"]

    base_transform = get_val_aug(image_size)

    tta_transforms = [
        base_transform,
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]

    augmented_images = []
    for transform in tta_transforms:
        if isinstance(image, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image

            pil_image = to_pil_image(image)
            augmented = transform(pil_image)
        else:
            augmented = transform(image)
        augmented_images.append(augmented)

    return augmented_images


def visualize_augmentations(image, config: dict, num_samples: int = 8):
    """Visualize augmentation effects for debugging."""
    import matplotlib.pyplot as plt
    import numpy as np

    transform = get_train_aug(config["data"]["image_size"])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        augmented = transform(image)

        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = augmented.transpose(1, 2, 0)
        augmented = augmented * std + mean
        augmented = np.clip(augmented, 0, 1)

        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmentation {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("report_assets/augmentation_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Augmentation samples saved to report_assets/augmentation_samples.png")
