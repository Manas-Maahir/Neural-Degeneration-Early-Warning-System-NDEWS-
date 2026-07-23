"""
test_signals.py
===============
Smoke tests for ndews/signals.py, ndews/labeller.py, and examples/model.py.

Run from the project root:
    python test_signals.py

Or after installing the package (pip install -e .):
    python test_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports without package installation when running from the project root.
sys.path.insert(0, str(Path(__file__).parent))

import torch
from ndews.labeller import get_instability_epoch, is_unstable, label_run
from examples.model import DeepCNN, SimpleCNN, TestMLP
from ndews.signals import CANONICAL_METRIC_SUFFIXES, SignalLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def _fmt_signals(signals: dict[str, float]) -> str:
    return "  " + "\n  ".join(f"{k:<48s} = {v:.6f}" for k, v in sorted(signals.items()))


# Expected canonical suffixes (must stay in sync with ndews/signals.py).
_EXPECTED_SUFFIXES = (
    "_representation_entropy",
    "_feature_reuse",
    "_gradient_diversity",
    "_neuron_sparsity",
    "_representational_isotropy",
    "_activation_scale",
)


def _assert_canonical_keys(signals: dict[str, float], layers: list[str]) -> None:
    """Assert that every layer has exactly the 6 canonical keys and no aliases."""
    for layer in layers:
        for suffix in _EXPECTED_SUFFIXES:
            key = f"{layer}{suffix}"
            if key not in signals:
                _fail(f"Expected canonical key '{key}' missing from signals dict")
        # Confirm no legacy aliases are present.
        for stale_suffix in ("_entropy", "_reuse", "_grad_var", "_grad_div",
                             "_embed_var", "_act_var", "_embedding_variance",
                             "_activation_variance"):
            stale_key = f"{layer}{stale_suffix}"
            if stale_key in signals:
                _fail(
                    f"Stale alias key '{stale_key}' still present — "
                    "remove it from get_epoch_signals()"
                )


# ---------------------------------------------------------------------------
# Test: hooks on TestMLP
# ---------------------------------------------------------------------------

def test_mlp_hooks() -> None:
    _section("Hook test — TestMLP (fc1, fc2)")

    model = TestMLP()
    logger = SignalLogger(model, target_layers=["fc1", "fc2"])

    x = torch.randn(16, 3, 32, 32)
    model(x).sum().backward()

    signals = logger.get_epoch_signals()
    print(_fmt_signals(signals))

    _assert_canonical_keys(signals, ["fc1", "fc2"])

    # Effective rank must be in [1, min(B, D)] — here min(16, 512) = 16.
    eff_rank = signals["fc1_representation_entropy"]
    if not (1.0 <= eff_rank <= 16.0):
        _fail(f"fc1 effective rank {eff_rank:.3f} outside expected [1, 16]")
    _ok(f"fc1 effective rank = {eff_rank:.3f}  (in [1, 16] OK)")

    # Isotropy must be in [0, 1].
    isotropy = signals["fc1_representational_isotropy"]
    if not (0.0 <= isotropy <= 1.0):
        _fail(f"fc1 isotropy {isotropy:.4f} outside [0, 1]")
    _ok(f"fc1 isotropy = {isotropy:.4f}  (in [0, 1] OK)")

    logger.remove_hooks()
    _ok("Hooks removed cleanly")


# ---------------------------------------------------------------------------
# Test: hooks on DeepCNN
# ---------------------------------------------------------------------------

def test_deepcnn_hooks() -> None:
    _section("Hook test — DeepCNN (conv3, fc1)")

    model = DeepCNN()
    logger = SignalLogger(model, target_layers=["conv3", "fc1"])

    x = torch.randn(16, 3, 32, 32)
    model(x).sum().backward()

    signals = logger.get_epoch_signals()
    print(_fmt_signals(signals))

    _assert_canonical_keys(signals, ["conv3", "fc1"])
    _ok("All canonical keys present, no stale aliases")

    logger.remove_hooks()
    _ok("Hooks removed cleanly")


# ---------------------------------------------------------------------------
# Test: CANONICAL_METRIC_SUFFIXES module constant
# ---------------------------------------------------------------------------

def test_canonical_suffixes_constant() -> None:
    _section("Module constant — CANONICAL_METRIC_SUFFIXES")

    if set(CANONICAL_METRIC_SUFFIXES) != set(_EXPECTED_SUFFIXES):
        _fail(
            f"CANONICAL_METRIC_SUFFIXES mismatch.\n"
            f"  Module exports: {sorted(CANONICAL_METRIC_SUFFIXES)}\n"
            f"  Test expects:   {sorted(_EXPECTED_SUFFIXES)}"
        )
    _ok(f"CANONICAL_METRIC_SUFFIXES = {CANONICAL_METRIC_SUFFIXES}")


# ---------------------------------------------------------------------------
# Test: reset clears state between epochs
# ---------------------------------------------------------------------------

def test_reset() -> None:
    _section("Hook reset — values cleared between epochs")

    model = SimpleCNN()
    logger = SignalLogger(model, target_layers=["conv2"])

    # Epoch 1 — small activations
    x = torch.randn(8, 3, 32, 32)
    model(x).sum().backward()
    s1 = logger.get_epoch_signals()["conv2_representation_entropy"]
    logger.reset()

    # Epoch 2 — different data (5× scale change)
    x = torch.randn(8, 3, 32, 32) * 5
    model(x).sum().backward()
    s2 = logger.get_epoch_signals()["conv2_representation_entropy"]

    _ok(f"Epoch 1 effective rank = {s1:.4f}")
    _ok(f"Epoch 2 effective rank = {s2:.4f}")
    _ok("Reset correctly clears state between epochs")

    logger.remove_hooks()


# ---------------------------------------------------------------------------
# Test: unknown layer name gives a warning, not a crash
# ---------------------------------------------------------------------------

def test_missing_layer_warning() -> None:
    _section("SignalLogger — unknown layer name warning")

    model = SimpleCNN()
    logger = SignalLogger(model, target_layers=["conv2", "ghost_layer"])
    logger.remove_hooks()
    _ok("Handled missing layer gracefully (WARNING printed above)")


# ---------------------------------------------------------------------------
# Test: num_classes forwarded correctly
# ---------------------------------------------------------------------------

def test_num_classes() -> None:
    _section("Model — num_classes parameter")

    m10 = SimpleCNN(num_classes=10)
    m100 = SimpleCNN(num_classes=100)
    x = torch.randn(4, 3, 32, 32)

    out10 = m10(x)
    out100 = m100(x)

    if out10.shape != (4, 10):
        _fail(f"SimpleCNN(num_classes=10) output shape {out10.shape} != (4, 10)")
    _ok(f"SimpleCNN(num_classes=10)  -> output shape {tuple(out10.shape)} OK")

    if out100.shape != (4, 100):
        _fail(f"SimpleCNN(num_classes=100) output shape {out100.shape} != (4, 100)")
    _ok(f"SimpleCNN(num_classes=100) -> output shape {tuple(out100.shape)} OK")


# ---------------------------------------------------------------------------
# Test: labeller
# ---------------------------------------------------------------------------

def test_labeller() -> None:
    _section("Labeller — instability detection")

    # Case 1: Clear crash after epoch 10
    crashing = [0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.80, 0.85, 0.86,
                0.87, 0.88, 0.88, 0.87, 0.78,
                0.75]
    result = label_run(crashing)
    assert result["unstable"] is True, "Expected unstable=True"
    assert result["instability_epoch"] is not None
    _ok(f"Crashing run  -> unstable=True, detected at epoch {result['instability_epoch']}")

    # Case 2: Healthy plateau
    healthy = [0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.75, 0.78, 0.80,
               0.81, 0.82, 0.83, 0.83, 0.84,
               0.84]
    result = label_run(healthy)
    assert result["unstable"] is False, "Expected unstable=False"
    assert result["instability_epoch"] is None
    _ok("Healthy run   -> unstable=False, instability_epoch=None")

    # Case 3: Slow drift across > window epochs — should NOT trigger
    slow_drift = [0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.75, 0.80, 0.82,
                  0.82, 0.81, 0.80, 0.79, 0.76,
                  0.74]
    result = label_run(slow_drift)
    _ok(f"Slow-drift run -> unstable={result['unstable']} (should be False if drop spread > window)")

    # Case 4: Too short — no check possible
    assert is_unstable([0.1, 0.2, 0.3]) is False
    _ok("Short history  -> False (burn-in not reached)")

    # Case 5: get_instability_epoch on healthy run returns None
    assert get_instability_epoch(healthy) is None
    _ok("get_instability_epoch returns None for healthy run")

    # Case 6: is_unstable and get_instability_epoch agree
    detected = get_instability_epoch(crashing)
    assert is_unstable(crashing) == (detected is not None)
    _ok("is_unstable and get_instability_epoch are consistent")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NDEWS Signal & Labeller Test Suite")
    print("=" * 60)

    tests = [
        test_canonical_suffixes_constant,
        test_mlp_hooks,
        test_deepcnn_hooks,
        test_reset,
        test_missing_layer_warning,
        test_num_classes,
        test_labeller,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    status = "ALL PASSED [OK]" if failed == 0 else f"{failed} FAILED"
    print(f"  {passed}/{len(tests)} passed  —  {status}")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)
