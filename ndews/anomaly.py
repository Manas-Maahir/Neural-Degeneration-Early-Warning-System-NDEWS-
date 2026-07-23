"""
ndews/anomaly.py
================
Self-baselined anomaly detection over the 6 canonical collapse signals.

No pretraining and no user-collected data: each run establishes its own baseline
during a short warmup window, then every subsequent epoch is scored with a
*directional* z-score against that frozen baseline.  Only drift in the
collapse-ward direction contributes, so the engine fires when a model's own
telemetry starts to degenerate — regardless of architecture or task.

Signal directions (which way indicates collapse):

    effective rank (entropy)      down    -> -1
    gradient diversity            down    -> -1
    representational isotropy     down    -> -1
    feature reuse                 up      -> +1
    neuron sparsity               up      -> +1
    activation scale              either  ->  0   (two-sided: explosion or vanishing)

Pure NumPy / stdlib — no training, fully architecture-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

# Canonical short signal names (entropy, gradient_diversity, ...) live in predictor.
from ndews.predictor import CANONICAL_METRIC_NAMES  # noqa: F401  (re-exported for callers)


# ---------------------------------------------------------------------------
# Direction map
# ---------------------------------------------------------------------------

# Sign of collapse-ward drift per canonical signal.  ``0`` means two-sided:
# any large deviation from baseline (in either direction) is treated as drift.
COLLAPSE_DIRECTIONS: dict[str, int] = {
    "entropy": -1,                      # effective rank collapsing toward 1
    "gradient_diversity": -1,           # training signal going monolithic
    "representational_isotropy": -1,    # variance collapsing onto few dims
    "feature_reuse": +1,                # features becoming redundant
    "neuron_sparsity": +1,              # units going dead
    "activation_scale": 0,              # explosion *or* vanishing
}

# Floor on the baseline std so a perfectly-constant warmup signal does not divide
# by zero.  Kept small; the ``min_signals`` gate guards against a lone jumpy signal.
_STD_FLOOR = 1e-8


def directional_zscore(name: str, value: float, mean: float, std: float) -> float:
    """
    Signed z-score where a **positive** result means collapse-ward drift.

    Non-finite values (a diverged run) are treated as max-severity drift (``+inf``),
    never a crash.  Signals not in :data:`COLLAPSE_DIRECTIONS` default to two-sided.
    """
    if not math.isfinite(value):
        return math.inf
    std = max(std, _STD_FLOOR)
    direction = COLLAPSE_DIRECTIONS.get(name, 0)
    if direction == 0:
        return abs(value - mean) / std
    return direction * (value - mean) / std


# ---------------------------------------------------------------------------
# Rolling (warmup-frozen) baseline
# ---------------------------------------------------------------------------

class RollingBaseline:
    """
    Per-signal baseline frozen after a warmup window.

    Collects the first ``baseline_epochs`` observations of each signal, then freezes
    a ``(mean, std)`` pair per signal.  ``is_ready`` is ``False`` during warmup — the
    engine raises no alerts until the baseline exists — and the baseline does **not**
    chase later drift once frozen (that is what makes the drift detectable).
    """

    def __init__(self, baseline_epochs: int) -> None:
        if baseline_epochs < 1:
            raise ValueError(f"baseline_epochs must be >= 1, got {baseline_epochs}")
        self.baseline_epochs = int(baseline_epochs)
        self._samples: list[dict[str, float]] = []
        self._stats: dict[str, tuple[float, float]] = {}

    @property
    def is_ready(self) -> bool:
        return len(self._samples) >= self.baseline_epochs

    def observe(self, aggregated: Mapping[str, float]) -> None:
        """Record one warmup epoch. No-op once the baseline is frozen."""
        if self.is_ready:
            return
        self._samples.append({str(k): float(v) for k, v in aggregated.items()})
        if self.is_ready:
            self._freeze()

    def _freeze(self) -> None:
        keys = {k for sample in self._samples for k in sample}
        for key in keys:
            arr = np.array(
                [s[key] for s in self._samples if key in s], dtype=float
            )
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                self._stats[key] = (0.0, 0.0)
            else:
                self._stats[key] = (float(arr.mean()), float(arr.std()))

    def stats(self, name: str) -> tuple[float, float] | None:
        """Return frozen ``(mean, std)`` for ``name``, or ``None`` if unseen."""
        return self._stats.get(name)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyResult:
    """Outcome of scoring one epoch against the baseline."""

    z_scores: dict[str, float]
    drifting_signals: list[str]
    alert: bool
    ready: bool


class AnomalyEngine:
    """
    Warmup baseline + directional z-score alerting.

    Call :meth:`score` once per epoch with the aggregated 6-signal dict. During
    warmup the value is folded into the baseline and a not-ready result is returned;
    afterwards each signal is z-scored and any that cross ``z_threshold`` in the
    collapse-ward direction are reported. An alert fires when at least
    ``min_signals`` signals drift.
    """

    def __init__(
        self,
        baseline_epochs: int = 5,
        z_threshold: float = 2.5,
        min_signals: int = 2,
    ) -> None:
        if min_signals < 1:
            raise ValueError(f"min_signals must be >= 1, got {min_signals}")
        self.baseline = RollingBaseline(baseline_epochs)
        self.z_threshold = float(z_threshold)
        self.min_signals = int(min_signals)

    @property
    def is_ready(self) -> bool:
        return self.baseline.is_ready

    def score(self, aggregated: Mapping[str, float]) -> AnomalyResult:
        """Score one epoch; observes into the baseline while still warming up."""
        if not self.baseline.is_ready:
            self.baseline.observe(aggregated)
            return AnomalyResult(z_scores={}, drifting_signals=[], alert=False, ready=False)

        z_scores: dict[str, float] = {}
        drifting: list[str] = []
        for name, value in aggregated.items():
            st = self.baseline.stats(name)
            if st is None:
                continue
            mean, std = st
            z = directional_zscore(name, float(value), mean, std)
            z_scores[name] = z
            if z >= self.z_threshold:
                drifting.append(name)

        alert = len(drifting) >= self.min_signals
        return AnomalyResult(
            z_scores=z_scores, drifting_signals=drifting, alert=alert, ready=True
        )
