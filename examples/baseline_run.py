"""
examples/baseline_run.py
========================
CLI runner for CIFAR-10 instability experiments, driven by the library's
``ndews.CollapseMonitor``.

This example *demonstrates the library* rather than re-implementing it: a single
``CollapseMonitor`` handles signal collection, aggregation, self-baselined anomaly
detection, and (optionally) the supervised online predictor. The runner keeps only the
CIFAR/CLI concerns — data, model, LR schedule, and the configurable val-accuracy labeller
that backs the ground-truth instability flags.

Run from the project root (works with or without `pip install -e .`):

Example usage:
    python examples/baseline_run.py --regime normal --epochs 20
    python examples/baseline_run.py --regime delayed_collapse --epochs 30
    python examples/baseline_run.py --regime high_learning_rate --epochs 20 \\
        --predictor-path ./output/predictor/random_forest.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so `ndews` and the `examples` package resolve
# whether or not the package was installed (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ndews import CollapseMonitor
from ndews.labeller import label_run
from ndews.seed_utils import seed_everything
from examples.dataset import get_cifar_loaders
from examples.regimes import ALL_REGIMES, get_regime_config, build_model, resolve_target_layers
from examples.train import eval_epoch, train_epoch


# ---------------------------------------------------------------------------
# Display helpers (local to this runner — not shared library logic)
# ---------------------------------------------------------------------------

def _format_layer_metrics(layer: str, signals: dict[str, float]) -> str:
    eff_rank  = signals.get(f"{layer}_representation_entropy",    0.0)
    grad_div  = signals.get(f"{layer}_gradient_diversity",        0.0)
    reuse     = signals.get(f"{layer}_feature_reuse",             0.0)
    sparsity  = signals.get(f"{layer}_neuron_sparsity",           0.0)
    isotropy  = signals.get(f"{layer}_representational_isotropy", 0.0)
    scale     = signals.get(f"{layer}_activation_scale",          0.0)
    return (
        f"{layer:<8s} "
        f"EffRank={eff_rank:7.3f} "
        f"GD={grad_div:9.6f} "
        f"FR={reuse:8.5f} "
        f"NS={sparsity:7.4f} "
        f"Iso={isotropy:7.4f} "
        f"Scale={scale:9.4f}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 CNN and stream internal degeneration metrics."
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="normal",
        choices=ALL_REGIMES,
        help="Training regime to use. See examples/regimes.py for definitions.",
    )
    parser.add_argument(
        "--model", type=str, default="simple", choices=["simple", "deep"],
    )
    parser.add_argument(
        "--target-layers", type=str, default=None,
        help="Comma-separated layer names to hook. Defaults to regime/model defaults.",
    )
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--weight-decay",  type=float, default=None)
    parser.add_argument("--label-noise",   type=float, default=None)
    parser.add_argument("--class-imbalance", type=float, default=None)
    parser.add_argument("--train-fraction",  type=float, default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument("--data-root",     type=str,   default="./data")
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--drop-threshold",  type=float, default=0.08)
    parser.add_argument("--drop-window",     type=int,   default=5)
    parser.add_argument("--burn-in",         type=int,   default=10)
    parser.add_argument("--sustain-epochs",  type=int,   default=3)
    parser.add_argument(
        "--chance-level", type=float, default=0.10,
        help="Flag runs stuck at/below this accuracy as unstable (CIFAR-10 chance=0.10). "
             "Set to a negative value to disable.",
    )
    # -- CollapseMonitor anomaly-engine knobs (self-baselined; no pretraining) --
    parser.add_argument(
        "--baseline-epochs", type=int, default=5,
        help="Warmup epochs the monitor freezes its per-signal baseline over.",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=2.5,
        help="Per-signal directional drift cutoff, in standard deviations.",
    )
    parser.add_argument(
        "--min-signals", type=int, default=2,
        help="How many signals must drift (collapse-ward) to raise an anomaly alert.",
    )
    parser.add_argument(
        "--predictor-path",
        type=str,
        default="./output/predictor/random_forest.pkl",
        help=(
            "Path to a trained predictor artifact. "
            "If present, prints online collapse probability each epoch."
        ),
    )
    parser.add_argument(
        "--disable-collapse-probability",
        action="store_true",
        help="Suppress online collapse probability output.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    regime_cfg = get_regime_config(args.regime)

    # CLI flags override regime defaults.
    epochs        = args.epochs        if args.epochs        is not None else regime_cfg.epochs
    lr            = args.lr            if args.lr            is not None else regime_cfg.lr
    weight_decay  = args.weight_decay  if args.weight_decay  is not None else regime_cfg.weight_decay
    label_noise   = args.label_noise   if args.label_noise   is not None else regime_cfg.label_noise
    class_imbalance = args.class_imbalance if args.class_imbalance is not None else regime_cfg.class_imbalance
    train_fraction  = args.train_fraction  if args.train_fraction  is not None else regime_cfg.train_fraction

    # Negative disables the at-chance check; otherwise flag runs stuck at chance.
    chance_level = args.chance_level if args.chance_level >= 0 else None

    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seed_everything(args.seed)

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

    model = build_model(args.model).to(device)
    target_layers = resolve_target_layers(args.model, args.target_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Resolve the optional supervised predictor. The monitor loads and manages it (its
    # sliding window + schema-mismatch handling); we only decide whether to hand it a path.
    predictor_arg: Path | None = None
    if not args.disable_collapse_probability:
        predictor_path = Path(args.predictor_path)
        if predictor_path.exists():
            predictor_arg = predictor_path
        else:
            print(
                f"[Predictor] Not found at {predictor_path}. "
                "Run `python examples/run_pipeline.py` first to train one."
            )

    val_history: list[float] = []

    def labeller_state() -> dict:
        return label_run(
            val_history,
            drop_threshold=args.drop_threshold,
            window=args.drop_window,
            burn_in=args.burn_in,
            sustain_epochs=args.sustain_epochs,
            chance_level=chance_level,
        )

    print("=" * 110)
    print("Training Instability Monitor (CIFAR-10) — powered by ndews.CollapseMonitor")
    print(
        f"regime={args.regime}  model={args.model}  device={device}  "
        f"epochs={epochs}  batch={args.batch_size}  lr={lr:.5f}  wd={weight_decay:.5f}"
    )
    print(
        f"dataset_mods: label_noise={label_noise:.2f}  "
        f"class_imbalance={class_imbalance:.2f}  train_fraction={train_fraction:.2f}"
    )
    print(
        f"anomaly: baseline_epochs={args.baseline_epochs}  "
        f"z_threshold={args.z_threshold}  min_signals={args.min_signals}"
    )
    if regime_cfg.lr_boost_at_epoch is not None:
        print(
            f"[LR schedule] LR will be multiplied by {regime_cfg.lr_boost_factor}× "
            f"at epoch {regime_cfg.lr_boost_at_epoch}"
        )
    print(f"tracked_layers={target_layers}")
    print("=" * 110)

    # The CollapseMonitor attaches hooks here and detaches them on context exit.
    with CollapseMonitor(
        model,
        layers=target_layers,
        baseline_epochs=args.baseline_epochs,
        z_threshold=args.z_threshold,
        min_signals=args.min_signals,
        predictor=predictor_arg,
    ) as monitor:
        has_predictor = monitor._predictor is not None
        for epoch in range(1, epochs + 1):
            # Apply scheduled LR boost (e.g. delayed_collapse regime).
            if (
                regime_cfg.lr_boost_at_epoch is not None
                and epoch == regime_cfg.lr_boost_at_epoch
            ):
                for pg in optimizer.param_groups:
                    pg["lr"] *= regime_cfg.lr_boost_factor
                boosted_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[LR Boost] Epoch {epoch}: lr → {boosted_lr:.6f} "
                    f"(×{regime_cfg.lr_boost_factor})"
                )

            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, show_progress=False
            )
            # The monitor's train_only guard means the eval forward pass below does not
            # pollute the epoch's signal buffers, so scoring after eval is safe.
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device, show_progress=False
            )

            report = monitor.on_epoch_end(val_acc=val_acc)
            agg = report.aggregated

            val_history.append(val_acc)
            collapse = labeller_state()
            peak_acc = max(val_history)

            progress_pct = (epoch / epochs) * 100.0
            print(f"Epoch {epoch}/{epochs} | {progress_pct:.0f}% done")
            print(f"Monitor Status         = {report.status}")
            print(f"Drifting Signals       = {report.drifting_signals or 'none'}")
            if not has_predictor:
                print("Collapse Probability   = N/A")
            elif report.probability is None:
                print("Collapse Probability   = warming_up")
            else:
                print(f"Collapse Probability   = {report.probability:.2f}")

            print(f"Effective Rank         = {agg['entropy']:.3f}")
            print(f"Gradient Diversity     = {agg['gradient_diversity']:.6f}")
            print(f"Feature Reuse          = {agg['feature_reuse']:.5f}")
            print(f"Neuron Sparsity        = {agg['neuron_sparsity']:.4f}")
            print(f"Repr. Isotropy         = {agg['representational_isotropy']:.4f}")
            print(f"Activation Scale       = {agg['activation_scale']:.4f}")
            print(f"Validation Accuracy    = {val_acc * 100:.2f}%")
            print(f"Train Loss             = {train_loss:.4f}")
            print(f"Validation Loss        = {val_loss:.4f}")
            print(f"Peak Val Accuracy      = {peak_acc * 100:.2f}%")
            print(
                f"Instability Flag       = {collapse['unstable']} "
                f"(instability_epoch={collapse['instability_epoch']})"
            )
            print("Layer-wise detail:")
            for layer in target_layers:
                print("    " + _format_layer_metrics(layer, report.signals))
            print("-" * 110)

    final_state = labeller_state()
    print("Run complete.")
    print(
        f"Final: unstable={final_state['unstable']}  "
        f"instability_epoch={final_state['instability_epoch']}  "
        f"best_val_acc={max(val_history) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
