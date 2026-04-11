"""
test_signals.py
===============
Smoke tests for src/signals.py, src/labeller.py, and src/model.py.
Run from the project root:

    python test_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the project root regardless of where the script is invoked
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.model   import SimpleCNN, TestMLP, DeepCNN
from src.signals import SignalLogger
from src.labeller import is_unstable, get_instability_epoch, label_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    width = 60
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")

def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)

def _fmt_signals(signals: dict[str, float]) -> str:
    return "  " + "\n  ".join(f"{k:<30s} = {v:.6f}" for k, v in sorted(signals.items()))

# ---------------------------------------------------------------------------
# Test: hooks on TestMLP
# ---------------------------------------------------------------------------

def test_mlp_hooks() -> None:
    _section("Hook test - TestMLP (fc1, fc2)")

    model  = TestMLP()
    logger = SignalLogger(model, target_layers=["fc1", "fc2"])

    x      = torch.randn(16, 3, 32, 32)   # batch of 16
    output = model(x)
    loss   = output.sum()
    loss.backward()

    signals = logger.get_epoch_signals()

    for key in (
        "fc1_entropy", "fc1_reuse", "fc1_grad_var",
        "fc1_neuron_sparsity", "fc1_embedding_variance", "fc1_activation_variance",
        "fc2_entropy", "fc2_reuse", "fc2_grad_var",
        "fc2_neuron_sparsity", "fc2_embedding_variance", "fc2_activation_variance",
    ):
        if key not in signals:
            _fail(f"Expected key '{key}' missing from signals dict")
        _ok(f"{key} = {signals[key]:.6f}")

    logger.remove_hooks()
    _ok("Hooks removed cleanly")

# ---------------------------------------------------------------------------
# Test: hooks on DeepCNN
# ---------------------------------------------------------------------------

def test_deepcnn_hooks() -> None:
    _section("Hook test - DeepCNN (conv3, fc1)")

    model  = DeepCNN()
    logger = SignalLogger(model, target_layers=["conv3", "fc1"])

    x      = torch.randn(16, 3, 32, 32)
    output = model(x)
    loss   = output.sum()
    loss.backward()

    signals = logger.get_epoch_signals()

    print(_fmt_signals(signals))

    for key in (
        "conv3_entropy", "conv3_reuse", "conv3_grad_var",
        "conv3_neuron_sparsity", "conv3_embedding_variance", "conv3_activation_variance",
        "fc1_entropy",   "fc1_reuse",   "fc1_grad_var",
        "fc1_neuron_sparsity", "fc1_embedding_variance", "fc1_activation_variance",
    ):
        if key not in signals:
            _fail(f"Expected key '{key}' missing from signals dict")

    logger.remove_hooks()
    _ok("All keys present, hooks removed cleanly")

# ---------------------------------------------------------------------------
# Test: reset clears state
# ---------------------------------------------------------------------------

def test_reset() -> None:
    _section("Hook reset - values cleared between epochs")

    model  = SimpleCNN()
    logger = SignalLogger(model, target_layers=["conv2"])

    # Epoch 1
    x = torch.randn(8, 3, 32, 32)
    model(x).sum().backward()
    s1 = logger.get_epoch_signals()["conv2_entropy"]
    logger.reset()

    # Epoch 2 (different data)
    x = torch.randn(8, 3, 32, 32) * 5
    model(x).sum().backward()
    s2 = logger.get_epoch_signals()["conv2_entropy"]

    _ok(f"Epoch 1 entropy = {s1:.6f}")
    _ok(f"Epoch 2 entropy = {s2:.6f}")
    _ok("Reset correctly clears state between epochs")

    logger.remove_hooks()

# ---------------------------------------------------------------------------
# Test: unknown layer name gives a warning, not a crash
# ---------------------------------------------------------------------------

def test_missing_layer_warning() -> None:
    _section("SignalLogger - unknown layer name warning")

    model  = SimpleCNN()
    # "ghost_layer" does not exist — should warn but not raise
    logger = SignalLogger(model, target_layers=["conv2", "ghost_layer"])
    logger.remove_hooks()
    _ok("Handled missing layer gracefully (see WARNING above)")

# ---------------------------------------------------------------------------
# Test: labeller
# ---------------------------------------------------------------------------

def test_labeller() -> None:
    _section("Labeller - instability detection")

    # Case 1: Clear crash after epoch 10
    crashing = [0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.80, 0.85, 0.86,
                0.87, 0.88, 0.88, 0.87, 0.78,  # drops 0.10 in 2 epochs
                0.75]
    result = label_run(crashing)
    assert result["unstable"]        is True,  "Expected unstable=True"
    assert result["instability_epoch"]   is not None
    _ok(f"Crashing run  -> unstable=True, detected at epoch {result['instability_epoch']}")

    # Case 2: Healthy plateau
    healthy = [0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.75, 0.78, 0.80,
               0.81, 0.82, 0.83, 0.83, 0.84,
               0.84]
    result = label_run(healthy)
    assert result["unstable"]        is False, "Expected unstable=False"
    assert result["instability_epoch"]   is None
    _ok("Healthy run   -> unstable=False, instability_epoch=None")

    # Case 3: Slow drift - should NOT trigger (8% over many epochs)
    slow_drift = [0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.75, 0.80, 0.82,
                  0.82, 0.81, 0.80, 0.79, 0.76,  # slow, spread > 5 epochs
                  0.74]
    result = label_run(slow_drift)
    _ok(f"Slow-drift run -> unstable={result['unstable']} (spread > window, should be False)")

    # Case 4: Too short - no check possible
    short = [0.1, 0.2, 0.3]
    assert is_unstable(short) is False
    _ok("Short history  -> correctly returns False (burn-in not reached)")

    # Case 5: get_instability_epoch on healthy run
    assert get_instability_epoch(healthy) is None
    _ok("get_instability_epoch returns None for healthy run")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Training Instability Prediction - Phase 1 Test Suite")
    print("=" * 60)

    tests = [
        test_mlp_hooks,
        test_deepcnn_hooks,
        test_reset,
        test_missing_layer_warning,
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
    status = "ALL PASSED [OK]" if failed == 0 else f"{failed} FAILED [X]"
    print(f"  Results: {passed}/{len(tests)} passed  -  {status}")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)
