"""
examples/model.py
=================
CNN and MLP architectures used in the Training Instability Prediction project.

Models
------
SimpleCNN  : Baseline 2-layer CNN (CIFAR-10/100, 32x32 input).
TestMLP    : Lightweight fully-connected model for quick hook testing.
DeepCNN    : Deeper 3-layer CNN for stress-testing signal extraction.

All image models accept ``num_classes`` so they can be used with datasets
other than CIFAR-10 (e.g. CIFAR-100 with num_classes=100).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Baseline 2-layer convolutional network for CIFAR-10/100.

    Architecture
    ------------
    conv1 (3->32)  -> ReLU -> MaxPool(2)  ->  [B, 32, 16, 16]
    conv2 (32->64) -> ReLU -> MaxPool(2)  ->  [B, 64,  8,  8]
    fc1   (4096->256) -> ReLU
    fc2   (256->num_classes)   -> logits

    Hook targets (recommended): ``"conv2"``, ``"fc1"``
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestMLP(nn.Module):
    """
    Minimal 2-layer MLP — used for fast unit-testing of hook infrastructure.

    Input is flattened from (B, 3, 32, 32) -> (B, 3072).

    Hook targets (recommended): ``"fc1"``
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeepCNN(nn.Module):
    """
    Deeper 3-layer CNN — used to verify that hooks scale across model depth.

    Architecture
    ------------
    conv1 (3-> 32)  -> ReLU -> MaxPool(2)  ->  [B,  32, 16, 16]
    conv2 (32-> 64) -> ReLU -> MaxPool(2)  ->  [B,  64,  8,  8]
    conv3 (64->128) -> ReLU -> MaxPool(2)  ->  [B, 128,  4,  4]
    fc1   (2048->256) -> ReLU
    fc2   (256->num_classes)   -> logits

    Hook targets (recommended): ``"conv3"``, ``"fc1"``
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
