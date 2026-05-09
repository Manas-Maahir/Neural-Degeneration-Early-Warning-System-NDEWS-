"""
run_pipeline.py
===============
End-to-end demonstration pipeline for Training Instability Prediction.

Walkthrough
-----------
1. Data generation  — Run one healthy and one unstable training regime.
2. Signal logging   — Internal neural telemetry is captured epoch-by-epoch.
3. CSV logging      — Metrics saved to ``output/metrics_log.csv``.
4. Labelling        — Instability epoch detected from validation accuracy history.
5. Predictor        — Random Forest trained on sliding windows of the signals.

IMPORTANT — evaluation note
----------------------------
This script trains and evaluates the predictor on the SAME two runs.
The printed accuracy is therefore an *in-sample* metric (the model has seen
the data it is scored on).  It is intended only as a quick smoke-test that
the pipeline executes end-to-end without errors.

For valid (held-out) evaluation, run:
    python experiments/run_many_regimes.py --runs-per-regime 5
and use the collected session CSVs with scripts/train_predictor.py
(leave-one-run-out cross-validation against held-out seeds).
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import get_cifar_loaders
from src.labeller import get_instability_epoch
from src.model import SimpleCNN
from src.predictor import Predictor, canonical_aggregate_features, create_sliding_windows
from src.seed_utils import seed_everything
from src.signals import SignalLogger
from src.train import eval_epoch, train_epoch

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

EPOCHS = 10          # Shortened for rapid smoke-test execution
BATCH_SIZE = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 3
FORECAST_HORIZON = 2
OUTPUT_DIR = Path("output")
PREDICTOR_PATH = OUTPUT_DIR / "predictor" / "random_forest.pkl"
METRICS_CSV = OUTPUT_DIR / "metrics_log.csv"


def run_training_experiment(
    regime_name: str,
    label_noise: float = 0.0,
) -> tuple[list[dict], list[float]]:
    print(f"\n=> Regime: {regime_name}  (device={DEVICE})")

    train_loader, val_loader = get_cifar_loaders(
        batch_size=BATCH_SIZE,
        label_noise=label_noise,
        train_fraction=0.1,   # 10 % sub-sample for speed
    )

    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    logger = SignalLogger(model, target_layers=["conv2", "fc1"])

    metrics_history: list[dict] = []
    val_acc_history: list[float] = []

    for epoch in range(EPOCHS):
        logger.reset()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, show_progress=False
        )
        signals = logger.get_epoch_signals()
        _, val_acc = eval_epoch(model, val_loader, criterion, DEVICE, show_progress=False)

        print(
            f"  Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        signals["epoch"] = epoch + 1
        signals["val_accuracy"] = val_acc
        signals["regime"] = regime_name

        metrics_history.append(signals)
        val_acc_history.append(val_acc)

    logger.remove_hooks()
    return metrics_history, val_acc_history


def main() -> None:
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Data generation
    # ------------------------------------------------------------------
    print("--- Phase 1: Data Generation ---")
    healthy_metrics, healthy_acc = run_training_experiment(
        "Healthy", label_noise=0.0
    )
    unstable_metrics, unstable_acc = run_training_experiment(
        "Unstable", label_noise=0.8
    )

    all_metrics = healthy_metrics + unstable_metrics

    # Save metrics CSV (overwrite — see module docstring for versioned alternative).
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        base_keys = ["regime", "epoch", "val_accuracy"]
        signal_keys = sorted(k for k in all_metrics[0] if k not in base_keys)
        fieldnames = base_keys + signal_keys
        with METRICS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[INFO] Metrics log → {METRICS_CSV}")

    # ------------------------------------------------------------------
    # Phase 2: Instability labelling
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Instability Labelling ---")
    # Low thresholds suit the shortened 10-epoch smoke-test runs.
    healthy_inst = get_instability_epoch(
        healthy_acc, drop_threshold=0.02, burn_in=2
    )
    unstable_inst = get_instability_epoch(
        unstable_acc, drop_threshold=0.02, burn_in=2
    )
    print(f"  Healthy  instability epoch: {healthy_inst}")
    print(f"  Unstable instability epoch: {unstable_inst}")

    # ------------------------------------------------------------------
    # Phase 3: Train Random Forest predictor
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Train Random Forest Predictor ---")

    def extract_features(arr: list[dict]) -> list[dict]:
        stripped = [
            {k: v for k, v in m.items() if k not in ("epoch", "val_accuracy", "regime")}
            for m in arr
        ]
        return [canonical_aggregate_features(m) for m in stripped]

    feat_h = extract_features(healthy_metrics)
    feat_u = extract_features(unstable_metrics)

    X_h, y_h, feature_keys = create_sliding_windows(
        feat_h,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        instability_epoch=healthy_inst,
        return_feature_keys=True,
    )
    X_u, y_u = create_sliding_windows(
        feat_u,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        instability_epoch=unstable_inst,
        feature_keys=feature_keys,
    )

    X_train = X_h + X_u
    y_train = y_h + y_u

    if not X_train:
        print("Not enough epochs to create sliding windows. Exiting.")
        return

    predictor = Predictor(
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        feature_keys=feature_keys,
    )
    predictor.train(X_train, y_train)
    saved_path = predictor.save(PREDICTOR_PATH)
    print(f"[INFO] Predictor saved → {saved_path}")

    # ------------------------------------------------------------------
    # In-sample evaluation (smoke-test only — not a valid result).
    # ------------------------------------------------------------------
    print(
        f"\n[EVAL] In-sample evaluation on {len(X_train)} training windows "
        "(not a held-out result — see module docstring):"
    )
    predictor.evaluate(X_train, y_train)
    print("\nPipeline smoke-test complete.")


if __name__ == "__main__":
    main()
