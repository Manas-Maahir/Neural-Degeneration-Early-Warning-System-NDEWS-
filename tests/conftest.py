"""
tests/conftest.py
=================
Shared fixtures for the ndews test suite.

Deliberately CIFAR-free: the library must be testable without the examples/ harness
or any dataset download, so we define tiny in-repo models here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Minimal MLP. Recommended hook layers: ``fc1`` (last-and-only) / ``fc2``."""

    IN_DIM = 48

    def __init__(self, hidden: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(self.IN_DIM, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyConvNet(nn.Module):
    """Minimal conv net. suggest_layers -> ``[conv2, fc]``. Any HxW via adaptive pool."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def tiny_mlp() -> TinyMLP:
    torch.manual_seed(0)
    return TinyMLP()


@pytest.fixture
def tiny_conv() -> TinyConvNet:
    torch.manual_seed(0)
    return TinyConvNet()


@pytest.fixture
def mlp_batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(16, TinyMLP.IN_DIM)


@pytest.fixture
def conv_batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(16, 3, 8, 8)
