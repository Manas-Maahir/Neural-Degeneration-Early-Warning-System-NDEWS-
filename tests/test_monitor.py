"""Tests for CollapseMonitor / MonitorReport (ndews/monitor.py)."""

from __future__ import annotations

import pytest
import torch

from ndews import CollapseMonitor, MonitorReport
from ndews.predictor import CANONICAL_METRIC_NAMES

BASE = {
    "entropy": 10.0,
    "gradient_diversity": 1.0,
    "feature_reuse": 0.1,
    "neuron_sparsity": 0.1,
    "representational_isotropy": 0.5,
    "activation_scale": 1.0,
}


def _feed(monitor, signal_seq, val_accs=None):
    """Drive on_epoch_end with monkeypatch-injected signal dicts."""
    it = iter(signal_seq)
    monitor._collect_signals = lambda: next(it)  # type: ignore[assignment]
    reports = []
    for i in range(len(signal_seq)):
        acc = None if val_accs is None else val_accs[i]
        reports.append(monitor.on_epoch_end(val_acc=acc))
    return reports


# --- lifecycle --------------------------------------------------------------

def test_hooks_attached_then_removed_on_close(tiny_mlp):
    monitor = CollapseMonitor(tiny_mlp, layers=["fc1"], baseline_epochs=2)
    assert len(monitor._logger._handles) > 0
    monitor.close()
    assert len(monitor._logger._handles) == 0


def test_context_manager_removes_hooks(tiny_mlp):
    with CollapseMonitor(tiny_mlp, layers=["fc1"]) as monitor:
        assert len(monitor._logger._handles) > 0
    assert len(monitor._logger._handles) == 0


def test_zero_resolved_layers_raises(tiny_mlp):
    with pytest.raises(ValueError):
        CollapseMonitor(tiny_mlp, layers=["does_not_exist"])


# --- warmup -> alert transition (synthetic signals) -------------------------

def test_warmup_then_alert(tiny_mlp):
    monitor = CollapseMonitor(
        tiny_mlp, layers=["fc1"], baseline_epochs=3, z_threshold=2.0, min_signals=2
    )
    collapsed = dict(BASE, entropy=1.0, feature_reuse=5.0)  # two collapse-ward
    reports = _feed(monitor, [BASE, BASE, BASE, collapsed])
    assert [r.status for r in reports[:3]] == ["warming_up"] * 3
    assert all(not r.alert for r in reports[:3])
    last = reports[3]
    assert last.alert and last.status == "warning"
    assert set(last.drifting_signals) == {"entropy", "feature_reuse"}
    monitor.close()


def test_min_signals_suppresses_single_drift(tiny_mlp):
    monitor = CollapseMonitor(
        tiny_mlp, layers=["fc1"], baseline_epochs=2, z_threshold=2.0, min_signals=2
    )
    one = dict(BASE, entropy=1.0)  # only one signal drifts
    reports = _feed(monitor, [BASE, BASE, one])
    assert reports[-1].drifting_signals == ["entropy"]
    assert not reports[-1].alert and reports[-1].status == "ok"
    monitor.close()


# --- report field population ------------------------------------------------

def test_report_fields_without_optionals(tiny_mlp):
    monitor = CollapseMonitor(tiny_mlp, layers=["fc1"], baseline_epochs=2)
    reports = _feed(monitor, [BASE, BASE])
    r = reports[-1]
    assert isinstance(r, MonitorReport)
    assert r.epoch == 2
    assert set(r.aggregated) == set(CANONICAL_METRIC_NAMES)
    assert r.probability is None          # no predictor supplied
    assert r.collapse_flag is None        # no val_acc supplied
    assert isinstance(r.message, str) and r.message
    monitor.close()


def test_collapse_flag_populated_with_val_acc(tiny_mlp):
    monitor = CollapseMonitor(tiny_mlp, layers=["fc1"], baseline_epochs=2)
    reports = _feed(monitor, [BASE, BASE, BASE], val_accs=[0.5, 0.6, 0.6])
    # A healthy val-acc history -> labeller confirms nothing, but the flag is a bool.
    assert reports[-1].collapse_flag is False
    monitor.close()


def test_status_collapse_requires_alert_and_labeller(tiny_mlp):
    monitor = CollapseMonitor(
        tiny_mlp, layers=["fc1"], baseline_epochs=3, z_threshold=2.0, min_signals=2
    )
    collapsed = dict(BASE, entropy=1.0, feature_reuse=5.0)
    # A val-acc series that crashes hard after a peak so the labeller fires.
    accs = [0.2, 0.4, 0.6, 0.8, 0.85, 0.86, 0.87, 0.88, 0.88, 0.87, 0.88, 0.20]
    signals = [BASE] * 11 + [collapsed]
    reports = _feed(monitor, signals, val_accs=accs)
    last = reports[-1]
    assert last.alert and last.collapse_flag is True and last.status == "collapse"
    monitor.close()


# --- optional supervised predictor ------------------------------------------

class _StubPredictor:
    window_size = 2
    forecast_horizon = 1
    feature_keys = list(CANONICAL_METRIC_NAMES)

    def predict_probability_from_window(self, window):
        assert len(window) == self.window_size
        return 0.9


def test_predictor_probability_warms_then_fires(tiny_mlp):
    monitor = CollapseMonitor(
        tiny_mlp, layers=["fc1"], baseline_epochs=1, predictor=_StubPredictor()
    )
    reports = _feed(monitor, [BASE, BASE, BASE])
    assert reports[0].probability is None            # window not full yet
    assert reports[1].probability == pytest.approx(0.9)
    assert reports[2].probability == pytest.approx(0.9)
    monitor.close()


# --- real forward pass end to end -------------------------------------------

def test_real_forward_pass_populates_signals(tiny_conv, conv_batch):
    monitor = CollapseMonitor(tiny_conv, baseline_epochs=2, min_signals=2)
    tiny_conv.train()
    tiny_conv(conv_batch).sum().backward()
    report = monitor.on_epoch_end(val_acc=0.5)
    assert set(report.aggregated) == set(CANONICAL_METRIC_NAMES)
    assert report.status == "warming_up"
    # Some raw signal was actually recorded from the training forward/backward.
    assert any(v != 0.0 for v in report.signals.values())
    monitor.close()
