"""Tests for the SignalLogger train_only guard (ndews/signals.py)."""

from __future__ import annotations

from ndews.signals import SignalLogger


def test_train_only_skips_eval_forward(tiny_mlp, mlp_batch):
    logger = SignalLogger(tiny_mlp, target_layers=["fc1"], train_only=True)

    tiny_mlp.eval()
    tiny_mlp(mlp_batch).sum().backward()
    eval_signals = logger.get_epoch_signals()
    assert all(v == 0.0 for v in eval_signals.values()), (
        "train_only hooks must record nothing in eval mode"
    )

    logger.reset()
    tiny_mlp.train()
    tiny_mlp(mlp_batch).sum().backward()
    train_signals = logger.get_epoch_signals()
    assert any(v != 0.0 for v in train_signals.values()), (
        "hooks must record in training mode"
    )

    logger.remove_hooks()


def test_train_only_false_records_in_eval(tiny_mlp, mlp_batch):
    logger = SignalLogger(tiny_mlp, target_layers=["fc1"], train_only=False)

    tiny_mlp.eval()
    tiny_mlp(mlp_batch).sum().backward()
    signals = logger.get_epoch_signals()
    assert any(v != 0.0 for v in signals.values()), (
        "train_only=False must record even in eval mode"
    )

    logger.remove_hooks()
