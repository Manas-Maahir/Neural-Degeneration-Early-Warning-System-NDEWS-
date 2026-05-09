"""
src/regimes.py
==============
Experiment regime definitions and model construction utilities.

This module is the single source of truth for:
- What a "regime" is (RegimeConfig dataclass)
- What regimes exist (REGIME_REGISTRY)
- How to build a model by name (build_model)
- How to resolve target hook layers (resolve_target_layers)

Previously this logic lived in experiments/baseline__run.py and was imported
by run_many_regimes.py via private-function imports — a coupling that broke
whenever baseline__run.py was refactored.  All experiment scripts now import
from here.

Regime notes
------------
normal               : Clean CIFAR-10 baseline.
label_noise          : 35 % random label corruption — accuracy ceiling, not
                       genuine instability (use as a contrast case only).
over_regularization  : High weight decay suppresses learning.
high_learning_rate   : Adam lr=0.05 causes instability relatively quickly.
overtraining         : Long run on full dataset — memorisation / overfitting.
class_imbalance      : 5 classes downsampled to 20 % keep-probability.
reduced_dataset_size : Only 20 % of training data retained.
delayed_collapse     : **Realistic instability scenario.**  Trains normally
                       for 10 epochs then applies a 50× LR multiplier.
                       Produces a run that looks healthy before it fails —
                       the primary test case for early-warning capability.
warm_then_overfit    : Long run with no regularisation; the model warms up
                       and then progressively memorises, causing val accuracy
                       to plateau then gently degrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from src.model import DeepCNN, SimpleCNN


# ---------------------------------------------------------------------------
# Regime configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """
    Full specification for one training experiment regime.

    Parameters
    ----------
    epochs : int
        Total number of training epochs.
    lr : float
        Initial learning rate for the Adam optimiser.
    weight_decay : float
        L2 regularisation coefficient.
    label_noise : float
        Fraction of training labels randomly replaced [0, 1].
    class_imbalance : float
        Keep-probability for ``imbalance_classes`` [0, 1].  1.0 = no effect.
    train_fraction : float
        Fraction of training data to retain after imbalance filtering [0, 1].
    lr_boost_at_epoch : int | None
        If set, the learning rate is multiplied by ``lr_boost_factor`` at the
        start of this epoch.  Used by ``delayed_collapse`` to simulate a
        training catastrophe after a healthy warm-up period.
    lr_boost_factor : float
        Multiplier applied when ``lr_boost_at_epoch`` is reached.
    """
    epochs: int
    lr: float
    weight_decay: float
    label_noise: float
    class_imbalance: float
    train_fraction: float
    lr_boost_at_epoch: int | None = field(default=None)
    lr_boost_factor: float = field(default=10.0)


# ---------------------------------------------------------------------------
# Regime registry
# ---------------------------------------------------------------------------

REGIME_REGISTRY: dict[str, RegimeConfig] = {
    # ---- standard baselines ----
    "normal": RegimeConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
    "label_noise": RegimeConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.35,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
    "over_regularization": RegimeConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=0.10,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
    "high_learning_rate": RegimeConfig(
        epochs=20,
        lr=0.05,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
    "overtraining": RegimeConfig(
        epochs=80,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
    "class_imbalance": RegimeConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=0.20,
        train_fraction=1.0,
    ),
    "reduced_dataset_size": RegimeConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=0.20,
    ),

    # ---- research-grade instability scenarios ----
    #
    # delayed_collapse
    # ----------------
    # Trains at lr=1e-3 for 10 epochs (healthy ascent), then multiplies
    # lr by 50× at epoch 11, causing gradient explosion / loss divergence.
    # This is the PRIMARY test case: the model is on-track when it fails,
    # so any predictor must use internal signals — not just "did training
    # start badly?" — to detect the impending catastrophe.
    "delayed_collapse": RegimeConfig(
        epochs=30,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
        lr_boost_at_epoch=11,
        lr_boost_factor=50.0,
    ),

    # warm_then_overfit
    # -----------------
    # Long run with no regularisation on full data.  The model warms up
    # cleanly, then progressively memorises training examples.  Val accuracy
    # plateaus and slowly degrades — a gentler instability than
    # delayed_collapse but representative of real production failures.
    "warm_then_overfit": RegimeConfig(
        epochs=60,
        lr=1e-3,
        weight_decay=0.0,
        label_noise=0.0,
        class_imbalance=1.0,
        train_fraction=1.0,
    ),
}

ALL_REGIMES: list[str] = list(REGIME_REGISTRY.keys())


def get_regime_config(name: str) -> RegimeConfig:
    """Return the ``RegimeConfig`` for *name*, or raise ``KeyError``."""
    if name not in REGIME_REGISTRY:
        raise KeyError(
            f"Unknown regime: {name!r}. "
            f"Available regimes: {sorted(REGIME_REGISTRY)}"
        )
    return REGIME_REGISTRY[name]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """
    Construct a model by name.

    Parameters
    ----------
    name : str
        ``"simple"`` → :class:`SimpleCNN`,  ``"deep"`` → :class:`DeepCNN`.
    num_classes : int
        Number of output logits.  Default 10 (CIFAR-10).
        Pass 100 for CIFAR-100 experiments.
    """
    if name == "deep":
        return DeepCNN(num_classes=num_classes)
    if name == "simple":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name!r}. Choose 'simple' or 'deep'.")


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

_DEFAULT_LAYERS: dict[str, list[str]] = {
    "simple": ["conv2", "fc1"],
    "deep":   ["conv3", "fc1"],
}


def resolve_target_layers(
    model_name: str,
    layers_arg: str | None,
) -> list[str]:
    """
    Return the list of layer names to hook.

    Parameters
    ----------
    model_name : str
        ``"simple"`` or ``"deep"``.
    layers_arg : str | None
        Comma-separated layer names from CLI (e.g. ``"conv2,fc1"``).
        ``None`` falls back to the defaults for *model_name*.
    """
    if layers_arg:
        return [s.strip() for s in layers_arg.split(",") if s.strip()]
    return _DEFAULT_LAYERS.get(model_name, ["fc1"])
