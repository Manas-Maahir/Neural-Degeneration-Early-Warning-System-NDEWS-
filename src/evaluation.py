"""
src/evaluation.py
=================
Leave-one-run-out cross-validation (LORO-CV) and baseline comparisons
for the instability predictor.

Design
------
- ``RunData``              dataclass holding per-run metrics + ground-truth
- ``leave_one_run_out_cv`` trains and evaluates the RF predictor via LORO-CV
- ``evaluate_baselines``   runs heuristic baselines on the same LORO splits
- ``MajorityClassBaseline``, ``ValAccDropBaseline``, ``RandomBaseline``
  produce window-level predictions without using learned signal features

Typical usage
-------------
    from src.evaluation import RunData, leave_one_run_out_cv, evaluate_baselines

    runs = [RunData(run_id, metrics_seq, val_accs, instability_epoch), ...]
    cv_result = leave_one_run_out_cv(runs, window_size=3, forecast_horizon=2)
    base_result = evaluate_baselines(runs, window_size=3, forecast_horizon=2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.predictor import (
    Predictor,
    canonical_aggregate_features,
    create_sliding_windows,
)


# ---------------------------------------------------------------------------
# RunData
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """
    Container for one training run's per-epoch signals and ground-truth label.

    Parameters
    ----------
    run_id : str
        Unique identifier (e.g. ``"delayed_collapse__seed100"``).
    metrics_sequence : list[dict[str, float]]
        One signal dict per epoch, produced by ``SignalLogger.get_epoch_signals()``.
    val_accuracies : list[float]
        Validation accuracy per epoch, parallel to ``metrics_sequence``.
    instability_epoch : int | None
        0-based epoch index of first detected instability.  ``None`` → stable run.
    """
    run_id: str
    metrics_sequence: list[dict[str, float]]
    val_accuracies: list[float]
    instability_epoch: int | None = field(default=None)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None,
) -> dict[str, float]:
    result: dict[str, float] = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            result["roc_auc"] = float("nan")
    else:
        result["roc_auc"] = float("nan")
    return result


def _aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Return mean ± std across folds for each metric key."""
    if not fold_metrics:
        return {}
    agg: dict[str, float] = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics if not math.isnan(m[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"]  = float("nan")
    return agg


# ---------------------------------------------------------------------------
# LORO-CV
# ---------------------------------------------------------------------------

def leave_one_run_out_cv(
    runs: list[RunData],
    *,
    window_size: int = 3,
    forecast_horizon: int = 2,
    predictor_kwargs: dict[str, Any] | None = None,
    aggregate_fn: Callable[[dict[str, float]], dict[str, float]] = canonical_aggregate_features,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Leave-one-run-out cross-validation for the instability predictor.

    For each run ``r``:
    - Train a fresh ``Predictor`` on windows from all other runs.
    - Evaluate on windows from ``r`` (held-out).

    Parameters
    ----------
    runs : list[RunData]
        At least 2 runs; needs at least one stable and one unstable for
        meaningful ROC-AUC.
    window_size, forecast_horizon : int
        Passed to ``create_sliding_windows``.
    predictor_kwargs : dict | None
        Extra kwargs forwarded to ``Predictor.__init__``.
    aggregate_fn : callable
        Feature aggregation function applied to each epoch's signal dict
        before windowing.  Defaults to ``canonical_aggregate_features``.
    verbose : bool
        Print per-fold summary to stdout.

    Returns
    -------
    dict with keys:
        ``fold_results``   — list of per-fold dicts (run_id, n_windows, metrics)
        ``aggregate``      — mean ± std across folds for each metric
        ``skipped_folds``  — run_ids where the held-out fold had no windows
        ``class_counts``   — {"positive": n, "negative": n} across all folds
    """
    if len(runs) < 2:
        raise ValueError(f"LORO-CV requires at least 2 runs, got {len(runs)}")

    predictor_kwargs = predictor_kwargs or {}
    fold_results: list[dict[str, Any]] = []
    skipped: list[str] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    # Pre-aggregate features so we do it once per run.
    agg_sequences: list[list[dict[str, float]]] = [
        [aggregate_fn(ep) for ep in run.metrics_sequence]
        for run in runs
    ]

    # Resolve a shared, stable feature key set from ALL runs (union).
    all_keys: set[str] = set()
    for seq in agg_sequences:
        for ep in seq:
            all_keys.update(ep.keys())
    feature_keys = sorted(all_keys)

    if verbose:
        print(f"[LORO-CV] {len(runs)} runs  |  feature_keys={feature_keys}")

    for hold_idx, held_run in enumerate(runs):
        # Build training windows from all other runs.
        X_train: list[list[float]] = []
        y_train: list[int] = []

        for i, run in enumerate(runs):
            if i == hold_idx:
                continue
            X_i, y_i = create_sliding_windows(
                agg_sequences[i],
                window_size=window_size,
                forecast_horizon=forecast_horizon,
                instability_epoch=run.instability_epoch,
                feature_keys=feature_keys,
            )
            X_train.extend(X_i)
            y_train.extend(y_i)

        if not X_train:
            if verbose:
                print(f"  [SKIP] {held_run.run_id}: training set has no windows")
            skipped.append(held_run.run_id)
            continue

        predictor = Predictor(
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            feature_keys=feature_keys,
            **predictor_kwargs,
        )
        predictor.train(X_train, y_train)

        # Evaluate on held-out run.
        X_test, y_test = create_sliding_windows(
            agg_sequences[hold_idx],
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            instability_epoch=held_run.instability_epoch,
            feature_keys=feature_keys,
        )

        if not X_test:
            if verbose:
                print(f"  [SKIP] {held_run.run_id}: held-out run too short for windows")
            skipped.append(held_run.run_id)
            continue

        y_pred = list(predictor.predict(X_test).tolist())
        y_prob = list(predictor.predict_proba(X_test).tolist())
        metrics = _compute_metrics(y_test, y_pred, y_prob)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold_results.append({
            "run_id":    held_run.run_id,
            "n_windows": len(y_test),
            "n_pos":     sum(y_test),
            **metrics,
        })

        if verbose:
            inst = held_run.instability_epoch
            print(
                f"  Fold {hold_idx+1:2d}/{len(runs)} "
                f"| {held_run.run_id:<40s} "
                f"| inst={inst!s:<6} "
                f"| n={len(y_test):3d} "
                f"| acc={metrics['accuracy']:.3f} "
                f"| f1={metrics['f1']:.3f} "
                f"| auc={metrics.get('roc_auc', float('nan')):.3f}"
            )

    aggregate = _aggregate_folds([f for f in fold_results])
    class_counts = {
        "positive": sum(all_y_true),
        "negative": len(all_y_true) - sum(all_y_true),
    }

    if verbose and fold_results:
        print(f"\n[LORO-CV] Aggregate over {len(fold_results)} folds:")
        for key, val in sorted(aggregate.items()):
            print(f"  {key:<28s} = {val:.4f}")
        print(f"  class_counts: {class_counts}")

    return {
        "fold_results":  fold_results,
        "aggregate":     aggregate,
        "skipped_folds": skipped,
        "class_counts":  class_counts,
    }


# ---------------------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------------------

class MajorityClassBaseline:
    """
    Always predicts the majority class from the training windows.

    Calibrated per fold (majority label in training split).
    """

    def __init__(self) -> None:
        self._majority: int = 0

    def fit(self, y_train: Sequence[int]) -> None:
        if not y_train:
            self._majority = 0
            return
        self._majority = 1 if sum(y_train) * 2 >= len(y_train) else 0

    def predict(self, n: int) -> list[int]:
        return [self._majority] * n

    def predict_proba(self, n: int) -> list[float]:
        return [float(self._majority)] * n


class ValAccDropBaseline:
    """
    Heuristic: predict instability when the val accuracy in the window has
    dropped more than ``drop_threshold`` from the window's peak.

    This simulates what a practitioner would do without signal features —
    watch for accuracy degradation and raise an alert.  It does NOT use any
    internal model signals, only the validation accuracy sequence.
    """

    def __init__(self, drop_threshold: float = 0.05) -> None:
        self.drop_threshold = drop_threshold

    def predict_windows(
        self,
        val_accuracies: list[float],
        *,
        window_size: int,
        forecast_horizon: int,
    ) -> list[int]:
        """
        Produce a predicted label for each sliding window position.

        Positive (1) when: ``peak_in_window − last_in_window > drop_threshold``.
        """
        n = len(val_accuracies)
        max_start = n - window_size - forecast_horizon + 1
        preds: list[int] = []
        for start in range(max_start):
            window_vals = val_accuracies[start : start + window_size]
            peak = max(window_vals)
            last = window_vals[-1]
            preds.append(int(peak - last > self.drop_threshold))
        return preds


class RandomBaseline:
    """
    Predicts class 1 at the training-set positive rate (random with calibrated prior).

    Useful for verifying that any classifier scores above chance.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._pos_rate: float = 0.5

    def fit(self, y_train: Sequence[int]) -> None:
        if not y_train:
            self._pos_rate = 0.5
            return
        self._pos_rate = sum(y_train) / len(y_train)

    def predict(self, n: int) -> list[int]:
        return [int(v) for v in (self._rng.random(n) < self._pos_rate)]

    def predict_proba(self, n: int) -> list[float]:
        return [float(self._pos_rate)] * n


# ---------------------------------------------------------------------------
# Baseline LORO evaluation
# ---------------------------------------------------------------------------

def evaluate_baselines(
    runs: list[RunData],
    *,
    window_size: int = 3,
    forecast_horizon: int = 2,
    val_acc_drop_threshold: float = 0.05,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all heuristic baselines using the same LORO splits.

    Returns a dict mapping baseline name → aggregate LORO metrics.

    Parameters
    ----------
    runs : list[RunData]
        Same list passed to ``leave_one_run_out_cv``.
    window_size, forecast_horizon : int
        Must match the values used in the predictor LORO evaluation.
    val_acc_drop_threshold : float
        Drop threshold for ``ValAccDropBaseline``.
    verbose : bool
        Print per-baseline aggregate metrics.
    """
    baselines: dict[str, Any] = {
        "majority_class": MajorityClassBaseline(),
        f"val_acc_drop_{val_acc_drop_threshold}": ValAccDropBaseline(val_acc_drop_threshold),
        "random": RandomBaseline(seed=0),
    }

    results: dict[str, dict[str, Any]] = {}

    for name, baseline in baselines.items():
        fold_metrics: list[dict[str, float]] = []

        for hold_idx, held_run in enumerate(runs):
            # Build ground-truth windows for held-out run.
            # We only need the y labels — use a dummy 1-feature sequence.
            dummy_seq = [{"_": 0.0}] * len(held_run.metrics_sequence)
            _, y_test = create_sliding_windows(
                dummy_seq,
                window_size=window_size,
                forecast_horizon=forecast_horizon,
                instability_epoch=held_run.instability_epoch,
                feature_keys=["_"],
            )
            if not y_test:
                continue

            if isinstance(baseline, ValAccDropBaseline):
                y_pred = baseline.predict_windows(
                    held_run.val_accuracies,
                    window_size=window_size,
                    forecast_horizon=forecast_horizon,
                )
                y_prob = [float(p) for p in y_pred]
            else:
                # Fit majority/random on training labels.
                train_y: list[int] = []
                for i, run in enumerate(runs):
                    if i == hold_idx:
                        continue
                    dummy_i = [{"_": 0.0}] * len(run.metrics_sequence)
                    _, y_i = create_sliding_windows(
                        dummy_i,
                        window_size=window_size,
                        forecast_horizon=forecast_horizon,
                        instability_epoch=run.instability_epoch,
                        feature_keys=["_"],
                    )
                    train_y.extend(y_i)

                baseline.fit(train_y)
                y_pred = baseline.predict(len(y_test))
                y_prob = baseline.predict_proba(len(y_test))

            metrics = _compute_metrics(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)

        agg = _aggregate_folds(fold_metrics)
        results[name] = {"fold_metrics": fold_metrics, "aggregate": agg}

        if verbose:
            print(f"[Baseline] {name}:")
            for key, val in sorted(agg.items()):
                print(f"  {key:<28s} = {val:.4f}")

    return results


# ---------------------------------------------------------------------------
# Convenience: print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    predictor_result: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
    metrics: Sequence[str] = ("accuracy_mean", "f1_mean", "roc_auc_mean"),
) -> None:
    """
    Print a side-by-side comparison of predictor vs. baselines.

    Parameters
    ----------
    predictor_result : dict
        Return value of ``leave_one_run_out_cv``.
    baseline_results : dict
        Return value of ``evaluate_baselines``.
    metrics : sequence of str
        Aggregate metric keys to display.
    """
    all_rows: list[tuple[str, dict[str, float]]] = [
        ("RF Predictor (LORO-CV)", predictor_result["aggregate"]),
    ]
    for name, res in baseline_results.items():
        all_rows.append((f"  Baseline: {name}", res["aggregate"]))

    col_w = 35
    header = f"{'Model':<{col_w}}" + "".join(f"{m:>18}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, agg in all_rows:
        row = f"{label:<{col_w}}"
        for m in metrics:
            val = agg.get(m, float("nan"))
            row += f"  {val:>7.4f}      " if not math.isnan(val) else f"  {'—':>7}      "
        print(row)
    print("=" * len(header))
