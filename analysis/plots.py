"""
analysis/plots.py
=================
Visualization utilities for instability experiment results.

All functions return the matplotlib Figure object so callers can save or
display it.  No interactive display is shown unless the caller calls
``plt.show()`` or ``fig.show()``.

Functions
---------
- ``plot_val_accuracy_curves``  — per-run val accuracy with instability markers
- ``plot_signal_trajectories``  — one or more signal metrics over epochs
- ``plot_lead_time_bar``        — lead-time results from signal_lag analysis
- ``plot_cv_metrics``           — per-fold LORO-CV metrics as a bar chart
- ``plot_feature_importance``   — RF feature importance from a trained Predictor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; caller sets backend if needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_REGIME_COLOURS: dict[str, str] = {
    "normal":               "#4CAF50",
    "label_noise":          "#F44336",
    "over_regularization":  "#FF9800",
    "high_learning_rate":   "#E91E63",
    "overtraining":         "#9C27B0",
    "class_imbalance":      "#2196F3",
    "reduced_dataset_size": "#00BCD4",
    "delayed_collapse":     "#FF5722",
    "warm_then_overfit":    "#795548",
}
_DEFAULT_STABLE_COLOUR   = "#4CAF50"
_DEFAULT_UNSTABLE_COLOUR = "#F44336"
_INSTABILITY_MARKER_COLOUR = "black"


def _regime_colour(regime: str, stable: bool) -> str:
    base = _REGIME_COLOURS.get(regime, "#607D8B")
    return base if not stable else _DEFAULT_STABLE_COLOUR


# ---------------------------------------------------------------------------
# Val accuracy curves
# ---------------------------------------------------------------------------

def plot_val_accuracy_curves(
    runs: list[dict[str, Any]],
    *,
    title: str = "Validation Accuracy over Training",
    figsize: tuple[float, float] = (10, 5),
    alpha: float = 0.8,
    mark_instability: bool = True,
) -> plt.Figure:
    """
    Plot validation accuracy curves for multiple runs.

    Parameters
    ----------
    runs : list of dicts with keys:
        ``run_id`` (str), ``val_accuracies`` (list[float]),
        ``instability_epoch`` (int | None), ``regime`` (str).
    mark_instability : bool
        If True, draw a vertical dashed line at ``instability_epoch`` for
        unstable runs.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for run in runs:
        accs = run["val_accuracies"]
        epochs = list(range(1, len(accs) + 1))
        inst = run.get("instability_epoch")
        regime = run.get("regime", "")
        stable = inst is None
        colour = _regime_colour(regime, stable)
        label = run.get("run_id", "")

        ax.plot(epochs, [a * 100 for a in accs],
                color=colour, alpha=alpha, linewidth=1.2, label=label)

        if mark_instability and inst is not None:
            ax.axvline(x=inst + 1, color=colour, linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(True, alpha=0.3)
    if len(runs) <= 12:
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Signal trajectories
# ---------------------------------------------------------------------------

def plot_signal_trajectories(
    epoch_signals: list[dict[str, float]],
    signal_keys: Sequence[str],
    *,
    instability_epoch: int | None = None,
    run_id: str = "",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot one or more signal metrics over epochs for a single run.

    Parameters
    ----------
    epoch_signals : list of dicts
        One dict per epoch from ``SignalLogger.get_epoch_signals()``.
    signal_keys : sequence of str
        Which keys to plot (e.g. ``["conv2_representation_entropy", ...]``).
    instability_epoch : int | None
        0-based epoch index at which instability was detected.  A vertical
        marker is drawn if provided.
    """
    n = len(signal_keys)
    if figsize is None:
        figsize = (10, 2.5 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    epochs = list(range(1, len(epoch_signals) + 1))

    for ax, key in zip(axes, signal_keys):
        values = [ep.get(key, float("nan")) for ep in epoch_signals]
        ax.plot(epochs, values, color="#1976D2", linewidth=1.3)
        ax.set_ylabel(key, fontsize=8)
        ax.grid(True, alpha=0.3)

        if instability_epoch is not None:
            ax.axvline(
                x=instability_epoch + 1,
                color=_INSTABILITY_MARKER_COLOUR,
                linestyle="--",
                linewidth=1.0,
                label="instability",
            )

    axes[-1].set_xlabel("Epoch")
    title = f"Signal Trajectories — {run_id}" if run_id else "Signal Trajectories"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Lead-time bar chart
# ---------------------------------------------------------------------------

def plot_lead_time_bar(
    lag_results: dict[str, dict[str, Any]],
    *,
    title: str = "Signal Lead-Time Before Collapse",
    figsize: tuple[float, float] | None = None,
    min_detection_rate: float = 0.0,
) -> plt.Figure:
    """
    Bar chart of median lead epochs for each signal, from signal_lag analysis.

    Parameters
    ----------
    lag_results : dict
        Return value of ``analysis.signal_lag.compute_signal_lags``.
    min_detection_rate : float
        Only show signals with detection rate >= this value.
    """
    import math

    filtered = {
        k: v for k, v in lag_results.items()
        if not math.isnan(v.get("lead_epochs", float("nan")))
        and v.get("detection_rate", 0.0) >= min_detection_rate
    }

    if not filtered:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    keys = sorted(filtered, key=lambda k: filtered[k]["lead_epochs"], reverse=True)
    leads = [filtered[k]["lead_epochs"] for k in keys]
    rates = [filtered[k]["detection_rate"] for k in keys]

    n = len(keys)
    if figsize is None:
        figsize = (10, max(4, n * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    colours = [
        "#1976D2" if r >= 0.7 else "#64B5F6" if r >= 0.4 else "#BBDEFB"
        for r in rates
    ]
    bars = ax.barh(range(n), leads, color=colours, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels([k.replace("_representation_entropy", "_eff_rank") for k in keys],
                       fontsize=8)
    ax.set_xlabel("Median Lead (epochs before collapse)")
    ax.set_title(title)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars with detection rate.
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(
            bar.get_width() + 0.05, i,
            f"{rate:.0%}",
            va="center", fontsize=7, color="#555",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LORO-CV per-fold metrics
# ---------------------------------------------------------------------------

def plot_cv_metrics(
    cv_result: dict[str, Any],
    *,
    metrics: Sequence[str] = ("accuracy", "f1", "roc_auc"),
    title: str = "LORO-CV Fold Metrics",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Bar chart of per-fold metrics from ``leave_one_run_out_cv``.

    Parameters
    ----------
    cv_result : dict
        Return value of ``ndews.evaluation.leave_one_run_out_cv``.
    metrics : sequence of str
        Which metric columns to plot.
    """
    fold_results = cv_result.get("fold_results", [])
    if not fold_results:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No fold results", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    run_ids = [f["run_id"] for f in fold_results]
    n_folds = len(run_ids)
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (max(10, n_folds * 0.6), 4 * n_metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]

    colours = ["#1976D2", "#388E3C", "#F57C00", "#C62828"]
    x = np.arange(n_folds)

    for ax, metric, colour in zip(axes, metrics, colours * 10):
        vals = [f.get(metric, float("nan")) for f in fold_results]
        ax.bar(x, vals, color=colour, alpha=0.8, edgecolor="white")

        # Mean line.
        agg = cv_result.get("aggregate", {})
        mean_val = agg.get(f"{metric}_mean")
        if mean_val is not None:
            ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.0,
                       label=f"mean={mean_val:.3f}")
            ax.legend(fontsize=8)

        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(run_ids, rotation=45, ha="right", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    predictor: Any,
    *,
    top_n: int = 20,
    title: str = "RF Feature Importance",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the trained RF predictor's feature importances.

    Parameters
    ----------
    predictor : Predictor
        A trained ``ndews.predictor.Predictor`` instance.
    top_n : int
        Show only the top N most important features.
    """
    rf = predictor.model
    feature_keys = predictor.feature_keys or []

    if not hasattr(rf, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Model has no feature_importances_ (non-RF?)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    importances: np.ndarray = rf.feature_importances_
    n_features = len(importances)
    window_size = int(getattr(predictor, "window_size", 1) or 1)

    if feature_keys and len(feature_keys) == n_features:
        labels = list(feature_keys)
    elif feature_keys and len(feature_keys) * window_size == n_features:
        # Windowed features are flattened step-outer, key-inner (see
        # predictor._flatten_window): index = step * n_keys + k. Label each with
        # its metric and how many epochs back in the window it came from.
        n_keys = len(feature_keys)
        labels = [
            f"{feature_keys[i % n_keys]} @t-{window_size - 1 - (i // n_keys)}"
            for i in range(n_features)
        ]
    else:
        labels = [f"feature_{i}" for i in range(n_features)]

    # Top N.
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_labels = [labels[i] for i in indices]

    n = len(top_labels)
    if figsize is None:
        figsize = (10, max(4, n * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
    ax.barh(range(n), top_importances[::-1], color=colours[::-1], edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Forecasting horizon sweep
# ---------------------------------------------------------------------------

def plot_horizon_sweep(
    sweep_result: dict[int, dict[str, Any]],
    *,
    metric: str = "roc_auc",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot forecasting skill vs lead time: RF predictor (internal signals) against
    the val-accuracy-drop baseline, over forecast horizons.

    Parameters
    ----------
    sweep_result : dict
        Return value of ``ndews.evaluation.forecast_horizon_sweep``.
    metric : str
        Base metric name, e.g. ``"roc_auc"``, ``"f1"``, ``"recall"`` — the
        ``"{metric}_mean"``/``"{metric}_std"`` keys are read from each aggregate.
    """
    horizons = sorted(sweep_result.keys())
    mean_key, std_key = f"{metric}_mean", f"{metric}_std"

    def _series(side: str, key: str) -> list[float]:
        return [sweep_result[h][side].get(key, float("nan")) for h in horizons]

    pred_mean = _series("predictor", mean_key)
    pred_std = _series("predictor", std_key)
    base_mean = _series("val_acc_drop", mean_key)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        horizons, pred_mean, yerr=pred_std, marker="o", capsize=3,
        color="#1976D2", linewidth=1.6, label="RF predictor (internal signals)",
    )
    ax.plot(
        horizons, base_mean, marker="s", linestyle="--",
        color="#F57C00", linewidth=1.4, label="val-accuracy-drop baseline",
    )
    if metric == "roc_auc":
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="chance")

    ax.set_xlabel("Forecast horizon (epochs ahead of collapse onset)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Forecasting skill vs lead time ({metric})")
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all figures to a directory
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib Figure and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
