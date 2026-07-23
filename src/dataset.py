"""
src/dataset.py
==============
CIFAR-10 data loading utilities with optional stress-regime controls.

Supported train-set perturbations:
- Label noise
- Class imbalance
- Reduced dataset size
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# CIFAR-10 per-channel mean/std for [-1, 1] normalisation
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)

# Windows does not support forked workers safely by default.
_DEFAULT_WORKERS = 0 if os.name == "nt" else 2


def _validate_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def _apply_label_noise(
    targets: list[int],
    noise_prob: float,
    n_classes: int,
    seed: int,
) -> list[int]:
    if noise_prob <= 0.0:
        return targets

    rng = random.Random(seed)
    noisy = list(targets)
    for i, old_label in enumerate(noisy):
        if rng.random() < noise_prob:
            # Sample a different class than the current label.
            candidate = rng.randrange(n_classes - 1)
            noisy[i] = candidate if candidate < old_label else candidate + 1
    return noisy


def _build_subset_indices(
    targets: list[int],
    class_imbalance: float,
    imbalance_classes: Iterable[int],
    train_fraction: float,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(len(targets)))

    if class_imbalance < 1.0:
        minority_set = set(imbalance_classes)
        kept: list[int] = []
        for idx in indices:
            label = int(targets[idx])
            if label in minority_set and rng.random() > class_imbalance:
                continue
            kept.append(idx)
        indices = kept

    if train_fraction < 1.0:
        subset_size = max(1, int(len(indices) * train_fraction))
        indices = rng.sample(indices, subset_size)
        indices.sort()

    return indices


def get_cifar_loaders(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: int = _DEFAULT_WORKERS,
    pin_memory: bool = True,
    *,
    label_noise: float = 0.0,
    class_imbalance: float = 1.0,
    imbalance_classes: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_fraction: float = 1.0,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader)`` for CIFAR-10.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for both loaders.
    data_root : str
        Directory where CIFAR-10 is downloaded / cached.
    num_workers : int
        DataLoader worker processes.
    pin_memory : bool
        Use pinned memory for faster GPU transfer when CUDA is available.
    label_noise : float
        Probability of random label replacement on train samples.
    class_imbalance : float
        Keep probability for classes listed in ``imbalance_classes``.
        Set to 1.0 for no class imbalance.
    imbalance_classes : tuple[int, ...]
        Class ids to downsample when ``class_imbalance < 1``.
    train_fraction : float
        Fraction of train data to retain after imbalance filtering.
    augment : bool
        Apply train-time augmentation (RandomCrop + RandomHorizontalFlip).
        Set ``False`` to study memorization/overfitting — augmentation
        suppresses the very degeneration these regimes are meant to induce.
    seed : int
        RNG seed used for deterministic perturbations and subsampling.
    """
    _validate_range("label_noise", label_noise, 0.0, 1.0)
    _validate_range("class_imbalance", class_imbalance, 0.0, 1.0)
    _validate_range("train_fraction", train_fraction, 0.0, 1.0)

    use_pin_memory = pin_memory and torch.cuda.is_available()

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    else:
        # No augmentation — let the model memorize the (small) train set.
        train_transform = val_transform

    train_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
    )

    # Apply label noise in-place on the base CIFAR targets list.
    train_targets = [int(t) for t in train_set.targets]
    train_set.targets = _apply_label_noise(
        train_targets,
        noise_prob=label_noise,
        n_classes=10,
        seed=seed,
    )

    # Build optional subset for class imbalance / reduced dataset size.
    subset_indices = _build_subset_indices(
        train_set.targets,
        class_imbalance=class_imbalance,
        imbalance_classes=imbalance_classes,
        train_fraction=train_fraction,
        seed=seed,
    )

    train_data = train_set
    if len(subset_indices) != len(train_set):
        train_data = Subset(train_set, subset_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
