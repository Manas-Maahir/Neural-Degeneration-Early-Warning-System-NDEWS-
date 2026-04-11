"""
run_pipeline.py
===============
End-to-end pipeline to showcase Training Instability Prediction.

Walkthrough:
1. Data Generation: Run healthy and unstable regimes on CIFAR-10. GPU accelerates this process. Number of epochs is lessened for rapid evaluation.
2. Signal Logging: The internal neural telemetry is captured throughout.
3. CSV Logging: Metrics appended and saved to `metrics_log.csv` for inspection.
4. Labelling: Calculate the 'instability epoch' using our specific objective metric constraints.
5. Random Forest Training: Train a sliding-window predictive ensemble model to classify/alert impending training instability in future steps.
"""

import os
import csv
import torch
import torch.nn as nn
from src.dataset import get_cifar_loaders
from src.model import SimpleCNN
from src.train import train_epoch, eval_epoch
from src.signals import SignalLogger
from src.labeller import get_instability_epoch
from src.predictor import create_sliding_windows, Predictor

EPOCHS = 10 # Shortened to speed up execution
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training_experiment(regime_name: str, label_noise: float = 0.0) -> tuple[list[dict], list[float]]:
    print(f"\n=> Starting Regime: {regime_name}")
    print(f"Using device: {DEVICE}")

    # Aggressive sub-sampling for extremely quick runs
    train_loader, val_loader = get_cifar_loaders(
        batch_size=BATCH_SIZE, label_noise=label_noise, train_fraction=0.1
    )

    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    logger = SignalLogger(model, target_layers=["conv2", "fc1"])

    metrics_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        logger.reset()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, show_progress=False)
        signals = logger.get_epoch_signals()
        
        _, val_acc = eval_epoch(model, val_loader, criterion, DEVICE, show_progress=False)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        signals["epoch"] = epoch + 1
        signals["val_accuracy"] = val_acc
        signals["regime"] = regime_name
        
        metrics_history.append(signals)
        val_acc_history.append(val_acc)

    logger.remove_hooks()
    return metrics_history, val_acc_history

def main():
    print("--- Phase 1: Data Generation ---")
    
    # Run 1: Healthy
    healthy_metrics, healthy_acc = run_training_experiment("Healthy", label_noise=0.0)
    
    # Run 2: Unstable (High Label Noise forces early plateau and drop)
    unstable_metrics, unstable_acc = run_training_experiment("Unstable", label_noise=0.8)

    all_metrics = healthy_metrics + unstable_metrics

    # Save metrics to log file
    csv_file = "metrics_log.csv"
    if all_metrics:
        # Move structural keys to the front
        base_keys = ["regime", "epoch", "val_accuracy"]
        signal_keys = sorted([k for k in all_metrics[0].keys() if k not in base_keys])
        fieldnames = base_keys + signal_keys
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[INFO] Saved comprehensive metrics log to {csv_file}")

    print("\n--- Phase 2: Instability Labelling ---")
    # Low burn-in and drop thresholds for shortened 10-epoch tracking
    healthy_inst = get_instability_epoch(healthy_acc, drop_threshold=0.02, burn_in=2)
    unstable_inst = get_instability_epoch(unstable_acc, drop_threshold=0.02, burn_in=2)
    
    print(f"Healthy regime instability epoch: {healthy_inst}")
    print(f"Unstable regime instability epoch: {unstable_inst}")

    print("\n--- Phase 3: Train Random Forest Predictor ---")
    
    # Filter out label-specific tracking logic from raw inference features
    def extract_features(arr):
        return [{k: v for k, v in m.items() if k not in ["epoch", "val_accuracy", "regime"]} for m in arr]
    
    feat_h = extract_features(healthy_metrics)
    feat_u = extract_features(unstable_metrics)
    
    X_h, y_h = create_sliding_windows(feat_h, window_size=3, forecast_horizon=2, instability_epoch=healthy_inst)
    X_u, y_u = create_sliding_windows(feat_u, window_size=3, forecast_horizon=2, instability_epoch=unstable_inst)

    X_train = X_h + X_u
    y_train = y_h + y_u

    if not X_train:
        print("Not enough epochs to create sliding windows.")
        return

    predictor = Predictor()
    predictor.train(X_train, y_train)

    # For proof-of-concept, evaluate on its own training set. In full scale, this tests over unseen seeds.
    print(f"\n[EVAL] Demonstrating evaluation over {len(X_train)} training windows:")
    predictor.evaluate(X_train, y_train)
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
