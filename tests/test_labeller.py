"""
Labeller tests (ndews/labeller.py).

Ported from the old top-level test_signals.py into pytest.
"""

from __future__ import annotations

from ndews.labeller import get_instability_epoch, is_unstable, label_run

_CRASHING = [0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.85, 0.86,
             0.87, 0.88, 0.88, 0.87, 0.78, 0.75]

_HEALTHY = [0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.75, 0.78, 0.80,
            0.81, 0.82, 0.83, 0.83, 0.84, 0.84]


def test_crash_after_peak_is_detected():
    result = label_run(_CRASHING)
    assert result["unstable"] is True
    assert result["instability_epoch"] is not None


def test_healthy_plateau_is_stable():
    result = label_run(_HEALTHY)
    assert result["unstable"] is False
    assert result["instability_epoch"] is None


def test_short_history_before_burn_in_is_stable():
    assert is_unstable([0.1, 0.2, 0.3]) is False


def test_is_unstable_agrees_with_get_instability_epoch():
    detected = get_instability_epoch(_CRASHING)
    assert is_unstable(_CRASHING) == (detected is not None)
    assert get_instability_epoch(_HEALTHY) is None


def test_stuck_at_chance_detected_via_chance_level():
    # Never learns: pinned near 10% chance for a 10-class task.
    stuck = [0.10, 0.11, 0.09, 0.10, 0.10, 0.11, 0.10, 0.09, 0.10, 0.10]
    assert is_unstable(stuck, chance_level=0.10) is True
    # Without a chance_level, drop-from-peak alone does not flag it.
    assert is_unstable(stuck, chance_level=None) is False
