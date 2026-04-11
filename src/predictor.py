"""
src/predictor.py
================
Predictive modeling to identify training instability early.
Using a Random Forest classifier over sliding windows of tracking metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_sliding_windows(
    run_metrics: list[dict[str, float]],
    window_size: int = 5,
    forecast_horizon: int = 3,
    instability_epoch: int | None = None
) -> tuple[list[list[float]], list[int]]:
    """
    Convert a list of epoch-level signal dictionaries into localized windows.
    Target y=1 if instability_epoch occurs within forecast_horizon from the END of the window.
    """
    if len(run_metrics) < window_size:
        return [], []

    X = []
    y = []

    # Get sorted keys to assure consistency across windows
    keys = sorted(run_metrics[0].keys())

    for idx in range(len(run_metrics) - window_size + 1):
        window_end_epoch = idx + window_size - 1

        # Flatten the window features temporally
        features = []
        for step in range(window_size):
            step_dict = run_metrics[idx + step]
            for k in keys:
                features.append(step_dict[k])
        
        # Label generation
        is_unstable_soon = 0
        if instability_epoch is not None:
            # If the instability happens within forecast_horizon after the current window
            if window_end_epoch < instability_epoch <= (window_end_epoch + forecast_horizon):
                is_unstable_soon = 1
        
        X.append(features)
        y.append(is_unstable_soon)

    return X, y

class Predictor:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
    def train(self, X_train: list[list[float]], y_train: list[int]):
        print(f"Training Random Forest on {len(X_train)} sliding windows...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: list[list[float]], y_test: list[int]) -> float:
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Evaluation Accuracy:", acc)
        if len(set(y_test)) > 1:
            print(classification_report(y_test, preds, zero_division=0))
        return acc

    def predict(self, X: list[list[float]]):
        return self.model.predict(X)
