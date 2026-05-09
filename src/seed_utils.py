"""
src/seed_utils.py
=================
Deterministic seeding for fully reproducible training runs.

Usage
-----
    from src.seed_utils import seed_everything
    seed_everything(42)

Notes
-----
- ``cudnn.deterministic = True`` disables non-deterministic cuDNN algorithms.
  This may reduce GPU throughput by ~10–20% on some operations.
- ``cudnn.benchmark = False`` prevents cuDNN from selecting the fastest
  algorithm based on input shape, which would reintroduce non-determinism.
- Call this function before constructing DataLoaders, models, or optimisers
  so that all random state is initialised from the same seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for deterministic reproduction.

    Parameters
    ----------
    seed : int
        The master seed. Every source of randomness is derived from this value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
