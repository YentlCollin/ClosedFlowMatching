"""Image dataset loaders for small-scale experiments (Colab-friendly)."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_image_dataloader(
    name: str = "mnist",
    batch_size: int = 128,
    image_size: int = 28,
    train: bool = True,
    n_samples: int | None = None,
) -> DataLoader:
    """Return a DataLoader for MNIST or Fashion-MNIST.

    Args:
        name: "mnist" or "fashion_mnist".
        batch_size: Batch size.
        image_size: Spatial resolution (images are resized to image_size x image_size).
        train: If True, load training split.
        n_samples: If set, subsample the dataset to this size.

    Returns:
        A DataLoader yielding (images, labels) with images in [-1, 1].
    """
    # TODO:
    # 1. Build a transforms.Compose that resizes, converts to tensor, and normalizes to [-1, 1]
    # 2. Load the appropriate torchvision dataset
    # 3. If n_samples is not None, use torch.utils.data.Subset
    # 4. Return a DataLoader with shuffle=True
    raise NotImplementedError


def flat_image_dim(image_size: int = 28, channels: int = 1) -> int:
    """Return the flattened dimension d = C * H * W."""
    return channels * image_size * image_size
