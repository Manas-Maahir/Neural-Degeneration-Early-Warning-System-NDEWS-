"""
ndews/monitor.py
================
`CollapseMonitor` — a model-agnostic, drop-in training monitor.

Wrap any ``torch.nn.Module`` training loop in three lines:

    from ndews import CollapseMonitor

    monitor = CollapseMonitor(model, layers=["layer3", "fc"])
    for epoch in range(epochs):
        train_one_epoch(...)              # model.train() -> hooks record
        acc = validate(...)               # model.eval()  -> hooks skip (train_only)
        report = monitor.on_epoch_end(val_acc=acc)
        if report.alert:
            print(report.status, report.drifting_signals, report.probability)
    monitor.close()                       # or use `with CollapseMonitor(...) as m:`

The default alert engine is self-baselined anomaly detection over the 6 internal
signals (see :mod:`ndews.anomaly`) — it needs no pretraining and no user data. An
optional supervised ``Predictor`` and the val-accuracy labeller layer on top when
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch.nn as nn

from ndews.anomaly import AnomalyEngine
from ndews.labeller import get_instability_epoch
from ndews.predictor import Predictor, canonical_aggregate_features
from ndews.signals import SignalLogger


# ---------------------------------------------------------------------------
# Layer suggestion
# ---------------------------------------------------------------------------

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def suggest_layers(model: nn.Module) -> list[str]:
    """
    Pick sensible default hook layers: the **last** conv-like module and the
    **last** linear module in ``model.named_modules()``.

    Prints its choice (never silent) so a caller who omitted ``layers`` sees what
    was instrumented. Returns 0, 1, or 2 names — an MLP yields just its last linear.
    """
    last_conv: str | None = None
    last_linear: str | None = None
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, _CONV_TYPES):
            last_conv = name
        elif isinstance(module, nn.Linear):
            last_linear = name

    chosen: list[str] = []
    for name in (last_conv, last_linear):
        if name is not None and name not in chosen:
            chosen.append(name)

    label = chosen if chosen else "no conv/linear layers found"
    print(f"[CollapseMonitor] suggest_layers selected {label} "
          "(last conv-like + last linear).")
    return chosen


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitorReport:
    """Immutable snapshot returned by :meth:`CollapseMonitor.on_epoch_end`."""

    epoch: int                              # 1-based epoch index
    signals: dict[str, float]               # raw per-layer signal dict (full detail)
    aggregated: dict[str, float]            # 6 canonical signals
    z_scores: dict[str, float]              # per-signal directional z vs baseline
    drifting_signals: list[str]             # signals past the threshold, collapse-ward
    alert: bool                             # True when >= min_signals drift
    status: str                             # warming_up / ok / warning / collapse
    probability: float | None               # optional supervised collapse proba
    collapse_flag: bool | None              # optional labeller confirmation
    message: str                            # one-line human-readable summary


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class CollapseMonitor:
    """
    Model-agnostic collapse monitor over the 6 NDEWS signals.

    Parameters
    ----------
    model : nn.Module
        The module being trained. Hooks are attached in ``__init__`` and removed by
        :meth:`close` (or the context manager).
    layers : list[str] | None
        Named modules to hook. ``None`` -> :func:`suggest_layers` (prints its pick).
    baseline_epochs : int
        Warmup epochs collected before anomaly scoring begins.
    z_threshold : float
        Per-signal directional drift cutoff, in standard deviations.
    min_signals : int
        How many signals must drift (collapse-ward) to raise an alert.
    predictor : str | Path | Predictor | None
        Optional supervised layer. A path is loaded via ``Predictor.load`` (its
        schema-mismatch warning is surfaced); on any load failure the monitor
        continues with anomaly-only alerts.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: list[str] | None = None,
        baseline_epochs: int = 5,
        z_threshold: float = 2.5,
        min_signals: int = 2,
        predictor: "str | Path | Predictor | None" = None,
    ) -> None:
        self.model = model
        if layers is None:
            layers = suggest_layers(model)

        self._logger = SignalLogger(model, target_layers=list(layers), train_only=True)
        if not self._logger._registered:
            self._logger.remove_hooks()
            raise ValueError(
                f"CollapseMonitor: none of the requested layers {list(layers)} "
                "resolved to a module in the model. Pass valid names from "
                "model.named_modules(), or layers=None to auto-select."
            )
        self.layers = sorted(self._logger._registered)

        self._baseline_epochs = int(baseline_epochs)
        self._engine = AnomalyEngine(baseline_epochs, z_threshold, min_signals)
        self._epoch = 0
        self._val_history: list[float] = []
        self._predictor = self._resolve_predictor(predictor)
        self._pred_window: list[dict[str, float]] = []

    # -- predictor plumbing --------------------------------------------------

    def _resolve_predictor(self, predictor):
        if predictor is None:
            return None
        if isinstance(predictor, (str, Path)):
            try:
                loaded = Predictor.load(predictor)
                print(
                    f"[CollapseMonitor] loaded predictor from {predictor} "
                    f"(window={loaded.window_size}, horizon={loaded.forecast_horizon})."
                )
                return loaded
            except Exception as exc:  # missing file, unpickling error, ...
                print(
                    f"[CollapseMonitor] failed to load predictor from {predictor!r}: "
                    f"{exc}. Continuing with anomaly-only alerts."
                )
                return None
        # Already a Predictor-like object (duck-typed).
        return predictor

    def _update_predictor(self, aggregated: Mapping[str, float]) -> float | None:
        if self._predictor is None:
            return None
        self._pred_window.append(dict(aggregated))
        window_size = self._predictor.window_size
        if len(self._pred_window) > window_size:
            self._pred_window = self._pred_window[-window_size:]
        if len(self._pred_window) < window_size:
            return None
        try:
            return float(self._predictor.predict_probability_from_window(self._pred_window))
        except Exception as exc:
            print(
                f"[CollapseMonitor] predictor inference failed: {exc}. "
                "Continuing with anomaly-only alerts."
            )
            return None

    # -- signal collection (test seam) --------------------------------------

    def _collect_signals(self) -> dict[str, float]:
        """Return this epoch's raw per-layer signals. Overridable seam for tests."""
        return self._logger.get_epoch_signals()

    # -- main entry point ----------------------------------------------------

    def on_epoch_end(self, val_acc: float | None = None) -> MonitorReport:
        """Score the epoch just finished and return a :class:`MonitorReport`."""
        self._epoch += 1
        signals = self._collect_signals()
        aggregated = canonical_aggregate_features(signals)

        probability = self._update_predictor(aggregated)
        result = self._engine.score(aggregated)

        if not result.ready:
            status = "warming_up"
        elif result.alert:
            status = "warning"
        else:
            status = "ok"

        collapse_flag: bool | None = None
        if val_acc is not None:
            self._val_history.append(float(val_acc))
            collapse_flag = get_instability_epoch(self._val_history) is not None
            if status == "warning" and collapse_flag:
                status = "collapse"

        # Reset hook buffers for the next epoch (no on_epoch_start needed).
        self._logger.reset()

        message = self._build_message(status, result, probability)
        return MonitorReport(
            epoch=self._epoch,
            signals=signals,
            aggregated=aggregated,
            z_scores=result.z_scores,
            drifting_signals=result.drifting_signals,
            alert=result.alert,
            status=status,
            probability=probability,
            collapse_flag=collapse_flag,
            message=message,
        )

    def _build_message(self, status: str, result, probability: float | None) -> str:
        if status == "warming_up":
            remaining = max(self._baseline_epochs - self._epoch, 0)
            return f"epoch {self._epoch}: warming up baseline ({remaining} epoch(s) left)"
        parts = [f"epoch {self._epoch}: {status}"]
        if result.drifting_signals:
            parts.append(f"drifting={result.drifting_signals}")
        if probability is not None:
            parts.append(f"p(collapse)={probability:.2f}")
        return " | ".join(parts)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Remove all hooks from the model."""
        self._logger.remove_hooks()

    def __enter__(self) -> "CollapseMonitor":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False
