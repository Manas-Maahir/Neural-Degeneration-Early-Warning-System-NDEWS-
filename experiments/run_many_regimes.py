"""
experiments/run_many_regimes.py
===============================
Batch runner for instability experiments.

This script runs multiple seeds across multiple regimes and stores:
1) Per-epoch CSV logs with internal metrics
2) A run-level summary CSV
3) A run manifest JSON with config metadata

Example:
    python experiments/run_many_regimes.py --runs-per-regime 3 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

# Allow direct execution from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.baseline__run import _build_model, _parse_layers, _regime_defaults
from src.dataset import get_cifar_loaders
from src.labeller import label_run
from src.signals import SignalLogger
from src.train import eval_epoch, train_epoch


CANONICAL_SUFFIXES = (
    "_representation_entropy",
    "_feature_reuse",
    "_gradient_diversity",
    "_neuron_sparsity",
    "_embedding_variance",
    "_activation_variance",
)

ALL_REGIMES = [
    "normal",
    "label_noise",
    "over_regularization",
    "high_learning_rate",
    "overtraining",
    "class_imbalance",
    "reduced_dataset_size",
]


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _canonical_signals(signals: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in signals.items()
        if any(key.endswith(suffix) for suffix in CANONICAL_SUFFIXES)
    }


def _ensure_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run many instability experiments and export per-epoch metric CSV files."
    )
    parser.add_argument("--regimes", type=str, default="all")
    parser.add_argument("--runs-per-regime", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "deep"])
    parser.add_argument("--target-layers", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./output/instability_runs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--drop-threshold", type=float, default=0.08)
    parser.add_argument("--drop-window", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--sustain-epochs", type=int, default=3)

    # Optional global overrides (applied on top of regime defaults).
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-noise", type=float, default=None)
    parser.add_argument("--class-imbalance", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.runs_per_regime < 1:
        raise ValueError(f"--runs-per-regime must be >= 1, got {args.runs_per_regime}")
    if args.sustain_epochs < 1:
        raise ValueError(f"--sustain-epochs must be >= 1, got {args.sustain_epochs}")

    regimes = ALL_REGIMES if args.regimes == "all" else _parse_csv_list(args.regimes)
    unknown = [r for r in regimes if r not in ALL_REGIMES]
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}. Known: {ALL_REGIMES}")

    if args.seeds:
        seeds = _parse_int_list(args.seeds)
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.runs_per_regime))
    if not seeds:
        raise ValueError("No seeds resolved. Provide --seeds or positive --runs-per-regime.")

    device = _resolve_device(args.device)
    output_root = Path(args.output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / f"session_{stamp}"
    runs_dir = session_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    target_layers = _parse_layers(args.target_layers, args.model)

    print("=" * 120)
    print("Batch Instability Experiment Runner")
    print(f"session={session_dir}")
    print(f"regimes={regimes}")
    print(f"seeds={seeds}")
    print(f"model={args.model} layers={target_layers} device={device}")
    print("=" * 120)

    summaries: list[dict[str, Any]] = []
    total_runs = len(regimes) * len(seeds)
    run_index = 0

    for regime in regimes:
        regime_defaults = _regime_defaults(regime)

        epochs = args.epochs if args.epochs is not None else regime_defaults.epochs
        lr = args.lr if args.lr is not None else regime_defaults.lr
        weight_decay = (
            args.weight_decay
            if args.weight_decay is not None
            else regime_defaults.weight_decay
        )
        label_noise = (
            args.label_noise if args.label_noise is not None else regime_defaults.label_noise
        )
        class_imbalance = (
            args.class_imbalance
            if args.class_imbalance is not None
            else regime_defaults.class_imbalance
        )
        train_fraction = (
            args.train_fraction
            if args.train_fraction is not None
            else regime_defaults.train_fraction
        )

        for seed in seeds:
            run_index += 1
            run_id = f"{regime}__seed{seed}"
            run_csv = runs_dir / f"{run_id}.csv"

            print(
                f"[{run_index:03d}/{total_runs:03d}] "
                f"run_id={run_id} epochs={epochs} lr={lr:.5f} wd={weight_decay:.5f} "
                f"noise={label_noise:.2f} imb={class_imbalance:.2f} frac={train_fraction:.2f}"
            )

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            val_history: list[float] = []
            rows: list[dict[str, Any]] = []

            try:
                train_loader, val_loader = get_cifar_loaders(
                    batch_size=args.batch_size,
                    data_root=args.data_root,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    label_noise=label_noise,
                    class_imbalance=class_imbalance,
                    train_fraction=train_fraction,
                    seed=seed,
                )

                model = _build_model(args.model).to(device)
                logger = SignalLogger(model, target_layers=target_layers)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                criterion = nn.CrossEntropyLoss()

                try:
                    for epoch in range(1, epochs + 1):
                        logger.reset()
                        train_loss = train_epoch(
                            model,
                            train_loader,
                            optimizer,
                            criterion,
                            device,
                            show_progress=False,
                        )
                        val_loss, val_acc = eval_epoch(
                            model,
                            val_loader,
                            criterion,
                            device,
                            show_progress=False,
                        )
                        val_history.append(val_acc)

                        collapse = label_run(
                            val_history,
                            drop_threshold=args.drop_threshold,
                            window=args.drop_window,
                            burn_in=args.burn_in,
                            sustain_epochs=args.sustain_epochs,
                        )

                        row: dict[str, Any] = {
                            "run_id": run_id,
                            "regime": regime,
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "peak_val_acc": max(val_history),
                            "unstable": bool(collapse["unstable"]),
                            "instability_epoch": collapse["instability_epoch"],
                            "model": args.model,
                            "target_layers": "|".join(target_layers),
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "label_noise": label_noise,
                            "class_imbalance": class_imbalance,
                            "train_fraction": train_fraction,
                            "drop_threshold": args.drop_threshold,
                            "drop_window": args.drop_window,
                            "burn_in": args.burn_in,
                            "sustain_epochs": args.sustain_epochs,
                        }
                        row.update(_canonical_signals(logger.get_epoch_signals()))
                        rows.append(row)

                    final_label = label_run(
                        val_history,
                        drop_threshold=args.drop_threshold,
                        window=args.drop_window,
                        burn_in=args.burn_in,
                        sustain_epochs=args.sustain_epochs,
                    )
                    _ensure_csv(run_csv, rows)
                    summary = {
                        "run_id": run_id,
                        "regime": regime,
                        "seed": seed,
                        "status": "ok",
                        "epochs": epochs,
                        "best_val_acc": max(val_history),
                        "final_val_acc": val_history[-1],
                        "unstable": bool(final_label["unstable"]),
                        "instability_epoch": final_label["instability_epoch"],
                        "csv_path": str(run_csv),
                        "error": "",
                    }
                    summaries.append(summary)
                    print(
                        f"    done best={summary['best_val_acc']*100:.2f}% "
                        f"final={summary['final_val_acc']*100:.2f}% "
                        f"unstable={summary['unstable']}"
                    )
                finally:
                    logger.remove_hooks()

            except Exception as exc:
                summary = {
                    "run_id": run_id,
                    "regime": regime,
                    "seed": seed,
                    "status": "failed",
                    "epochs": epochs,
                    "best_val_acc": "",
                    "final_val_acc": "",
                    "unstable": "",
                    "instability_epoch": "",
                    "csv_path": str(run_csv),
                    "error": str(exc),
                }
                summaries.append(summary)
                print(f"    failed: {exc}")

    summary_csv = session_dir / "run_summary.csv"
    _ensure_csv(summary_csv, summaries)

    manifest = {
        "session_dir": str(session_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "regimes": regimes,
            "seeds": seeds,
            "model": args.model,
            "target_layers": target_layers,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "data_root": args.data_root,
            "device": str(device),
            "drop_threshold": args.drop_threshold,
            "drop_window": args.drop_window,
            "burn_in": args.burn_in,
            "sustain_epochs": args.sustain_epochs,
            "overrides": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_noise": args.label_noise,
                "class_imbalance": args.class_imbalance,
                "train_fraction": args.train_fraction,
            },
        },
        "totals": {
            "runs_requested": total_runs,
            "runs_ok": sum(1 for s in summaries if s["status"] == "ok"),
            "runs_failed": sum(1 for s in summaries if s["status"] == "failed"),
        },
    }
    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 120)
    print(f"Summary CSV: {summary_csv}")
    print(f"Manifest   : {manifest_path}")
    print(
        f"Completed: ok={manifest['totals']['runs_ok']} "
        f"failed={manifest['totals']['runs_failed']}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
