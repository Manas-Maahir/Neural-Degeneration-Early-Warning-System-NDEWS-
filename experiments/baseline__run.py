"""
experiments/baseline__run.py
============================
CLI runner for CIFAR-10 instability experiments with live metric logging.

Example:
    python experiments/baseline__run.py --regime high_learning_rate --epochs 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn

# Allow direct execution: `python experiments/baseline__run.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_cifar_loaders
from src.labeller import label_run
from src.model import DeepCNN, SimpleCNN
from src.signals import SignalLogger
from src.train import eval_epoch, train_epoch


@dataclass
class RegimeConfig:
    epochs: int
    lr: float
    weight_decay: float
    label_noise: float
    class_imbalance: float
    train_fraction: float


def _regime_defaults(regime: str) -> RegimeConfig:
    defaults: dict[str, RegimeConfig] = {
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
    }
    return defaults[regime]


def _parse_layers(layers_arg: str | None, model_name: str) -> list[str]:
    if layers_arg:
        return [s.strip() for s in layers_arg.split(",") if s.strip()]
    return ["conv3", "fc1"] if model_name == "deep" else ["conv2", "fc1"]


def _build_model(model_name: str) -> nn.Module:
    if model_name == "deep":
        return DeepCNN()
    return SimpleCNN()


def _format_layer_metrics(layer: str, signals: dict[str, float]) -> str:
    entropy = signals.get(f"{layer}_representation_entropy", 0.0)
    grad_div = signals.get(f"{layer}_gradient_diversity", 0.0)
    reuse = signals.get(f"{layer}_feature_reuse", 0.0)
    sparsity = signals.get(f"{layer}_neuron_sparsity", 0.0)
    emb_var = signals.get(f"{layer}_embedding_variance", 0.0)
    act_var = signals.get(f"{layer}_activation_variance", 0.0)
    return (
        f"{layer:<8s} "
        f"H={entropy:7.4f} "
        f"GD={grad_div:9.6f} "
        f"FR={reuse:8.5f} "
        f"NS={sparsity:7.4f} "
        f"EV={emb_var:9.6f} "
        f"AV={act_var:9.6f}"
    )


def _aggregate_signal_metrics(
    target_layers: list[str],
    signals: dict[str, float],
) -> dict[str, float]:
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    entropy = [
        signals.get(f"{layer}_representation_entropy", 0.0) for layer in target_layers
    ]
    grad_div = [signals.get(f"{layer}_gradient_diversity", 0.0) for layer in target_layers]
    reuse = [signals.get(f"{layer}_feature_reuse", 0.0) for layer in target_layers]
    sparsity = [signals.get(f"{layer}_neuron_sparsity", 0.0) for layer in target_layers]
    emb_var = [signals.get(f"{layer}_embedding_variance", 0.0) for layer in target_layers]
    act_var = [signals.get(f"{layer}_activation_variance", 0.0) for layer in target_layers]

    return {
        "entropy": _mean(entropy),
        "gradient_diversity": _mean(grad_div),
        "feature_reuse": _mean(reuse),
        "neuron_sparsity": _mean(sparsity),
        "embedding_variance": _mean(emb_var),
        "activation_variance": _mean(act_var),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 CNN and stream internal degeneration metrics."
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="normal",
        choices=[
            "normal",
            "label_noise",
            "over_regularization",
            "high_learning_rate",
            "overtraining",
            "class_imbalance",
            "reduced_dataset_size",
        ],
    )
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "deep"])
    parser.add_argument("--target-layers", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-noise", type=float, default=None)
    parser.add_argument("--class-imbalance", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--drop-threshold", type=float, default=0.08)
    parser.add_argument("--drop-window", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--sustain-epochs", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    regime_cfg = _regime_defaults(args.regime)

    # CLI overrides on top of regime defaults.
    epochs = args.epochs if args.epochs is not None else regime_cfg.epochs
    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}")
    lr = args.lr if args.lr is not None else regime_cfg.lr
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else regime_cfg.weight_decay
    )
    label_noise = args.label_noise if args.label_noise is not None else regime_cfg.label_noise
    class_imbalance = (
        args.class_imbalance
        if args.class_imbalance is not None
        else regime_cfg.class_imbalance
    )
    train_fraction = (
        args.train_fraction if args.train_fraction is not None else regime_cfg.train_fraction
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader = get_cifar_loaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        pin_memory=True,
        label_noise=label_noise,
        class_imbalance=class_imbalance,
        train_fraction=train_fraction,
        seed=args.seed,
    )

    model = _build_model(args.model).to(device)
    target_layers = _parse_layers(args.target_layers, args.model)
    logger = SignalLogger(model, target_layers=target_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    val_history: list[float] = []

    print("=" * 110)
    print("Training Instability Monitor (CIFAR-10)")
    print(
        f"regime={args.regime} model={args.model} device={device} "
        f"epochs={epochs} batch={args.batch_size} lr={lr:.5f} wd={weight_decay:.5f}"
    )
    print(
        f"dataset_mods: label_noise={label_noise:.2f} "
        f"class_imbalance={class_imbalance:.2f} train_fraction={train_fraction:.2f}"
    )
    print(f"tracked_layers={target_layers}")
    print("=" * 110)

    try:
        for epoch in range(1, epochs + 1):
            logger.reset()
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, show_progress=False
            )
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device, show_progress=False
            )
            signals = logger.get_epoch_signals()

            val_history.append(val_acc)
            collapse = label_run(
                val_history,
                drop_threshold=args.drop_threshold,
                window=args.drop_window,
                burn_in=args.burn_in,
                sustain_epochs=args.sustain_epochs,
            )
            peak_acc = max(val_history)
            agg = _aggregate_signal_metrics(target_layers, signals)

            print(f"Epoch {epoch}")
            print(f"Entropy = {agg['entropy']:.2f}")
            print(f"Gradient Diversity = {agg['gradient_diversity']:.2f}")
            print(f"Feature Reuse = {agg['feature_reuse']:.2f}")
            print(f"Neuron Sparsity = {agg['neuron_sparsity']:.2f}")
            print(f"Embedding Variance = {agg['embedding_variance']:.2f}")
            print(f"Activation Variance = {agg['activation_variance']:.2f}")
            print(f"Validation Accuracy = {val_acc * 100:.2f}%")
            print(f"Train Loss = {train_loss:.4f}")
            print(f"Validation Loss = {val_loss:.4f}")
            print(f"Peak Validation Accuracy = {peak_acc * 100:.2f}%")
            print(
                f"Instability Flag = {collapse['unstable']} "
                f"(instability_epoch={collapse['instability_epoch']})"
            )
            print("Layer-wise detail:")
            for layer in target_layers:
                print("    " + _format_layer_metrics(layer, signals))
            print("-" * 110)
    finally:
        logger.remove_hooks()

    final_state = label_run(
        val_history,
        drop_threshold=args.drop_threshold,
        window=args.drop_window,
        burn_in=args.burn_in,
        sustain_epochs=args.sustain_epochs,
    )
    print("Run complete.")
    print(
        f"Final: unstable={final_state['unstable']} "
        f"instability_epoch={final_state['instability_epoch']} "
        f"best_val_acc={max(val_history) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
