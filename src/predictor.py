"""
src/predictor.py
================
Sliding-window feature utilities and lightweight instability predictor.

This module is used in two places:
1) Offline training (scripts/train_predictor.py, run_pipeline.py) to build
   a predictor from prior collected runs.
2) Online monitoring (experiments/baseline__run.py) to print per-epoch
   collapse probability while training is still in progress.

Signal name alignment
---------------------
The canonical metric names used here match the key suffixes produced by
``SignalLogger.get_epoch_signals()`` in src/signals.py.  If you rename a
hook or add a new one, update ``CANONICAL_METRIC_NAMES`` and
``_SUFFIX_TO_CANONICAL`` here accordingly.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# ---------------------------------------------------------------------------
# Canonical metric schema
# ---------------------------------------------------------------------------

CANONICAL_METRIC_NAMES: tuple[str, ...] = (
    "entropy",
    "gradient_diversity",
    "feature_reuse",
    "neuron_sparsity",
    "representational_isotropy",
    "activation_scale",
)

# Maps the signal key suffix (from SignalLogger) → canonical short name.
# Both the long canonical form (e.g. "_representation_entropy") and the
# short form (e.g. "entropy") are handled by canonical_aggregate_features.
_SUFFIX_TO_CANONICAL: dict[str, str] = {
    "_representation_entropy":      "entropy",
    "_gradient_diversity":          "gradient_diversity",
    "_feature_reuse":               "feature_reuse",
    "_neuron_sparsity":             "neuron_sparsity",
    "_representational_isotropy":   "representational_isotropy",
    "_activation_scale":            "activation_scale",
}

# Schema version — increment when the feature schema changes incompatibly.
_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------

def canonical_aggregate_features(signals: Mapping[str, float]) -> dict[str, float]:
    """
    Aggregate per-layer signal keys into a canonical feature vector.

    Averages values across all tracked layers for each metric type.
    Input keys may be either:
    - Short canonical names directly (e.g. ``"entropy"``), or
    - Layer-prefixed canonical suffixes (e.g. ``"conv2_representation_entropy"``).

    Returns a dict with exactly the keys in ``CANONICAL_METRIC_NAMES``.
    Missing metrics default to 0.0.
    """
    buckets: dict[str, list[float]] = {name: [] for name in CANONICAL_METRIC_NAMES}

    for key, value in signals.items():
        # Direct canonical name match (e.g. already aggregated upstream).
        if key in buckets:
            buckets[key].append(float(value))
            continue
        # Layer-prefixed key — strip layer name prefix using suffix matching.
        for suffix, canonical_name in _SUFFIX_TO_CANONICAL.items():
            if key.endswith(suffix):
                buckets[canonical_name].append(float(value))
                break

    return {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in buckets.items()
    }


# ---------------------------------------------------------------------------
# Sliding-window construction
# ---------------------------------------------------------------------------

def _resolve_feature_keys(
    metrics_sequence: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str] | None,
) -> list[str]:
    if feature_keys is not None:
        return [str(k) for k in feature_keys]
    keys: set[str] = set()
    for m in metrics_sequence:
        keys.update(str(k) for k in m.keys())
    return sorted(keys)


def _flatten_window(
    window_metrics: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
) -> list[float]:
    vec: list[float] = []
    for step in window_metrics:
        for key in feature_keys:
            vec.append(float(step.get(key, 0.0)))
    return vec


def create_sliding_windows(
    metrics_sequence: Sequence[Mapping[str, float]],
    *,
    window_size: int = 3,
    forecast_horizon: int = 2,
    instability_epoch: int | None = None,
    feature_keys: Sequence[str] | None = None,
    return_feature_keys: bool = False,
):
    """
    Convert epoch-wise metrics into supervised windows.

    A window ending at epoch ``t`` is labelled positive when instability first
    appears within the next ``forecast_horizon`` epochs, i.e. in ``(t, t+h]``.

    Parameters
    ----------
    metrics_sequence : sequence of dicts
        One dict per epoch.  Keys are metric names; values are floats.
    window_size : int
        Number of past epochs in each feature window.
    forecast_horizon : int
        Number of future epochs to check for the instability label.
    instability_epoch : int | None
        Epoch index (0-based) of first detected instability.  ``None`` means
        the run was stable — all windows are labelled 0.
    feature_keys : sequence of str | None
        Ordered list of keys to extract from each step.  If ``None``, inferred
        from the union of all keys in ``metrics_sequence`` (sorted).
    return_feature_keys : bool
        When ``True``, also return the resolved feature key list as the third
        element of the tuple.

    Returns
    -------
    (X, y) or (X, y, keys)
        X : list of flat float vectors, length = n_windows.
        y : list of int labels (0 or 1).
        keys : list[str] — only when ``return_feature_keys=True``.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")

    if len(metrics_sequence) < window_size + forecast_horizon:
        keys = _resolve_feature_keys(metrics_sequence, feature_keys)
        if return_feature_keys:
            return [], [], keys
        return [], []

    keys = _resolve_feature_keys(metrics_sequence, feature_keys)
    X: list[list[float]] = []
    y: list[int] = []

    max_start = len(metrics_sequence) - window_size - forecast_horizon + 1
    for start in range(max_start):
        end = start + window_size - 1
        future_end = end + forecast_horizon

        window = metrics_sequence[start : start + window_size]
        X.append(_flatten_window(window, keys))

        positive = (
            instability_epoch is not None
            and end < instability_epoch <= future_end
        )
        y.append(int(positive))

    if return_feature_keys:
        return X, y, keys
    return X, y


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Random-forest predictor wrapper with robust save/load and proba API."""

    def __init__(
        self,
        *,
        window_size: int = 3,
        forecast_horizon: int = 2,
        feature_keys: Sequence[str] | None = None,
        model=None,
    ) -> None:
        self.window_size = int(window_size)
        self.forecast_horizon = int(forecast_horizon)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.model = model or RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_keys: Sequence[str] | None = None,
    ) -> None:
        if not X:
            raise ValueError("Cannot train predictor: X is empty.")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

        if feature_keys is not None:
            self.feature_keys = [str(k) for k in feature_keys]

        labels = [int(v) for v in y]
        unique = sorted(set(labels))

        if len(unique) == 1:
            # Low-data edge case: all windows belong to one class.
            # Caller should be warned — a DummyClassifier is not a predictor.
            print(
                f"[Predictor] WARNING: only class {unique[0]} in training data. "
                "Falling back to DummyClassifier. Collect more runs with "
                "both stable and unstable examples."
            )
            self.model = DummyClassifier(strategy="constant", constant=unique[0])

        self.model.fit(X, labels)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return probability of class ``1`` (incoming instability)."""
        proba = self.model.predict_proba(X)
        classes = [int(c) for c in getattr(self.model, "classes_", [0, 1])]

        if len(classes) == 1:
            val = 1.0 if classes[0] == 1 else 0.0
            return np.full(len(X), val, dtype=float)

        if 1 not in classes:
            return np.zeros(len(X), dtype=float)

        pos_index = classes.index(1)
        return proba[:, pos_index]

    def predict_probability_from_window(
        self,
        window_metrics: Sequence[Mapping[str, float]],
    ) -> float:
        """Compute collapse probability from a raw window of per-epoch signal dicts."""
        if self.feature_keys is None:
            raise ValueError("predictor.feature_keys is not set.")
        if len(window_metrics) != self.window_size:
            raise ValueError(
                f"Expected {self.window_size} steps, got {len(window_metrics)}."
            )
        x = _flatten_window(window_metrics, self.feature_keys)
        return float(self.predict_proba([x])[0])

    def evaluate(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> dict[str, float]:
        preds = self.predict(X)
        probs = self.predict_proba(X)
        acc = float(accuracy_score(y, preds))
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y, preds, digits=4))

        metrics: dict[str, float] = {"accuracy": acc}
        try:
            auc = float(roc_auc_score(y, probs))
            print(f"ROC-AUC: {auc:.4f}")
            metrics["roc_auc"] = auc
        except ValueError:
            print("ROC-AUC: undefined (only one class present in y_true).")
        return metrics

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "model": self.model,
            "window_size": self.window_size,
            "forecast_horizon": self.forecast_horizon,
            "feature_keys": self.feature_keys,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        saved_version = payload.get("schema_version", 1)
        if saved_version != _SCHEMA_VERSION:
            print(
                f"[Predictor] WARNING: saved schema version {saved_version} "
                f"!= current {_SCHEMA_VERSION}. Feature keys may be stale. "
                "Re-train the predictor with the current codebase."
            )

        return cls(
            window_size=int(payload["window_size"]),
            forecast_horizon=int(payload["forecast_horizon"]),
            feature_keys=payload.get("feature_keys"),
            model=payload["model"],
        )
