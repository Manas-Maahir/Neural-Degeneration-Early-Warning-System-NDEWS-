"""
Tests for create_sliding_windows labelling (ndews/predictor.py).

The optional supervised path depends on this; ``forecast`` and ``detect`` modes had
no direct coverage before.
"""

from __future__ import annotations

import pytest

from ndews.predictor import create_sliding_windows

# One scalar feature per epoch; length 8. window_size=3, horizon=2 -> 4 windows,
# with start indices 0..3 and window-end indices 2..5.
SEQ = [{"m": float(i)} for i in range(8)]


def test_forecast_mode_positive_strictly_before_onset():
    # onset=5: positive when window-end < 5 <= end+2  -> starts 1 and 2.
    _, y = create_sliding_windows(
        SEQ, window_size=3, forecast_horizon=2, instability_epoch=5, label_mode="forecast"
    )
    assert y == [0, 1, 1, 0]


def test_detect_mode_includes_in_window_onset():
    # onset=5: positive when 5 <= end+2 -> starts 1,2,3; no aftermath windows here.
    _, y = create_sliding_windows(
        SEQ, window_size=3, forecast_horizon=2, instability_epoch=5, label_mode="detect"
    )
    assert y == [0, 1, 1, 1]


def test_detect_mode_drops_pure_aftermath_windows():
    # onset=1: windows starting after the onset (starts 2,3) are dropped entirely.
    X, y = create_sliding_windows(
        SEQ, window_size=3, forecast_horizon=2, instability_epoch=1, label_mode="detect"
    )
    assert len(X) == 2 and y == [1, 1]


def test_stable_run_is_all_negative():
    for mode in ("forecast", "detect"):
        _, y = create_sliding_windows(
            SEQ, window_size=3, forecast_horizon=2, instability_epoch=None, label_mode=mode
        )
        assert y == [0, 0, 0, 0]


def test_too_short_sequence_returns_empty():
    short = SEQ[:4]  # len 4 < window_size + horizon = 5
    X, y = create_sliding_windows(short, window_size=3, forecast_horizon=2)
    assert X == [] and y == []


def test_feature_keys_control_flatten_order():
    seq = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}]
    X, _, keys = create_sliding_windows(
        seq, window_size=2, forecast_horizon=1, feature_keys=["b", "a"],
        return_feature_keys=True,
    )
    assert keys == ["b", "a"]
    # One window (start 0), flattened step-major: [b0, a0, b1, a1].
    assert X[0] == [2.0, 1.0, 4.0, 3.0]


def test_invalid_label_mode_raises():
    with pytest.raises(ValueError):
        create_sliding_windows(SEQ, label_mode="nonsense")
