"""
Hook-infrastructure tests (ndews/signals.py).

Ported from the old top-level test_signals.py into pytest, on tiny in-repo models.
"""

from __future__ import annotations

from ndews.signals import CANONICAL_METRIC_SUFFIXES, SignalLogger

_STALE_SUFFIXES = (
    "_entropy", "_reuse", "_grad_var", "_grad_div",
    "_embed_var", "_act_var", "_embedding_variance", "_activation_variance",
)


def _train_step(model, x):
    model.train()
    model(x).sum().backward()


def test_canonical_keys_present_no_stale_aliases(tiny_mlp, mlp_batch):
    logger = SignalLogger(tiny_mlp, target_layers=["fc1", "fc2"])
    _train_step(tiny_mlp, mlp_batch)
    signals = logger.get_epoch_signals()

    for layer in ("fc1", "fc2"):
        for suffix in CANONICAL_METRIC_SUFFIXES:
            assert f"{layer}{suffix}" in signals
        for stale in _STALE_SUFFIXES:
            assert f"{layer}{stale}" not in signals

    logger.remove_hooks()


def test_effective_rank_and_isotropy_ranges(tiny_mlp, mlp_batch):
    logger = SignalLogger(tiny_mlp, target_layers=["fc1"])
    _train_step(tiny_mlp, mlp_batch)
    signals = logger.get_epoch_signals()

    eff_rank = signals["fc1_representation_entropy"]
    assert 1.0 <= eff_rank <= mlp_batch.shape[0]  # [1, min(B, D)], B=16 <= D=32

    isotropy = signals["fc1_representational_isotropy"]
    assert 0.0 <= isotropy <= 1.0

    logger.remove_hooks()


def test_reset_clears_between_epochs(tiny_conv, conv_batch):
    logger = SignalLogger(tiny_conv, target_layers=["conv2"])
    _train_step(tiny_conv, conv_batch)
    assert logger.get_epoch_signals()["conv2_activation_scale"] != 0.0

    logger.reset()
    # After reset, before any new pass, metrics are back to the empty default.
    assert logger.get_epoch_signals()["conv2_activation_scale"] == 0.0

    logger.remove_hooks()


def test_missing_layer_is_skipped_not_fatal(tiny_mlp):
    logger = SignalLogger(tiny_mlp, target_layers=["fc1", "ghost_layer"])
    assert "fc1" in logger._registered
    assert "ghost_layer" not in logger._registered
    logger.remove_hooks()


def test_canonical_suffixes_constant_shape():
    assert len(CANONICAL_METRIC_SUFFIXES) == 6
    assert "_representation_entropy" in CANONICAL_METRIC_SUFFIXES
