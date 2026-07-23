"""Tests for the self-baselined anomaly engine (ndews/anomaly.py)."""

from __future__ import annotations

import math

from ndews.anomaly import (
    AnomalyEngine,
    RollingBaseline,
    directional_zscore,
)

BASE = {
    "entropy": 10.0,
    "gradient_diversity": 1.0,
    "feature_reuse": 0.1,
    "neuron_sparsity": 0.1,
    "representational_isotropy": 0.5,
    "activation_scale": 1.0,
}


def _with(**changes) -> dict[str, float]:
    d = dict(BASE)
    d.update(changes)
    return d


# --- directional z-score ----------------------------------------------------

def test_directional_zscore_signs():
    # up-is-bad (feature_reuse, +1): above baseline is collapse-ward (positive).
    assert directional_zscore("feature_reuse", 3.0, 1.0, 1.0) == 2.0
    assert directional_zscore("feature_reuse", -1.0, 1.0, 1.0) == -2.0
    # down-is-bad (entropy, -1): below baseline is collapse-ward (positive).
    assert directional_zscore("entropy", 8.0, 10.0, 1.0) == 2.0
    assert directional_zscore("entropy", 12.0, 10.0, 1.0) == -2.0
    # two-sided (activation_scale, 0): any deviation is positive.
    assert directional_zscore("activation_scale", 3.0, 1.0, 1.0) == 2.0
    assert directional_zscore("activation_scale", -1.0, 1.0, 1.0) == 2.0


def test_nonfinite_is_max_severity():
    assert directional_zscore("entropy", float("nan"), 10.0, 1.0) == math.inf
    assert directional_zscore("feature_reuse", float("inf"), 1.0, 1.0) == math.inf


# --- rolling baseline -------------------------------------------------------

def test_rolling_baseline_freezes_after_warmup():
    b = RollingBaseline(3)
    assert not b.is_ready
    b.observe({"a": 1.0})
    b.observe({"a": 3.0})
    assert not b.is_ready
    b.observe({"a": 2.0})
    assert b.is_ready
    mean, std = b.stats("a")
    assert mean == 2.0 and std > 0
    # Once frozen, further observations are ignored (baseline does not chase drift).
    b.observe({"a": 100.0})
    assert b.stats("a")[0] == 2.0


def test_rolling_baseline_rejects_bad_warmup():
    import pytest

    with pytest.raises(ValueError):
        RollingBaseline(0)


# --- engine warmup / scoring ------------------------------------------------

def test_warmup_raises_no_alerts_then_scores():
    eng = AnomalyEngine(baseline_epochs=3, z_threshold=2.0, min_signals=2)
    for _ in range(3):
        r = eng.score(BASE)
        assert not r.ready and not r.alert and r.drifting_signals == []
    assert eng.is_ready
    # Identical to baseline -> no drift.
    r = eng.score(BASE)
    assert r.ready and not r.alert and r.drifting_signals == []


def test_collapse_ward_drift_vs_safe_direction():
    eng = AnomalyEngine(3, z_threshold=2.0, min_signals=1)
    for _ in range(3):
        eng.score(BASE)
    # entropy DOWN is collapse-ward -> drift.
    assert "entropy" in eng.score(_with(entropy=1.0)).drifting_signals

    eng2 = AnomalyEngine(3, z_threshold=2.0, min_signals=1)
    for _ in range(3):
        eng2.score(BASE)
    # entropy UP is the safe direction -> not drift.
    assert "entropy" not in eng2.score(_with(entropy=50.0)).drifting_signals


def test_min_signals_gate():
    eng = AnomalyEngine(3, z_threshold=2.0, min_signals=2)
    for _ in range(3):
        eng.score(BASE)
    one = eng.score(_with(entropy=1.0))  # single collapse-ward signal
    assert one.drifting_signals == ["entropy"] and not one.alert

    eng2 = AnomalyEngine(3, z_threshold=2.0, min_signals=2)
    for _ in range(3):
        eng2.score(BASE)
    two = eng2.score(_with(entropy=1.0, feature_reuse=5.0))  # two collapse-ward
    assert set(two.drifting_signals) == {"entropy", "feature_reuse"} and two.alert


def test_activation_scale_is_two_sided():
    up = AnomalyEngine(3, 2.0, 1)
    for _ in range(3):
        up.score(BASE)
    assert "activation_scale" in up.score(_with(activation_scale=5.0)).drifting_signals

    down = AnomalyEngine(3, 2.0, 1)
    for _ in range(3):
        down.score(BASE)
    assert "activation_scale" in down.score(_with(activation_scale=0.001)).drifting_signals


def test_nonfinite_signal_alerts_via_engine():
    eng = AnomalyEngine(2, z_threshold=2.0, min_signals=1)
    for _ in range(2):
        eng.score(BASE)
    r = eng.score(_with(entropy=float("nan")))
    assert "entropy" in r.drifting_signals and r.alert
