"""
src/labeller.py
===============
Objective instability detection for training runs.

Detection criterion:
- Start checking after ``burn_in`` epochs.
- Track running peak validation accuracy.
- If accuracy drops by more than ``drop_threshold`` from the latest peak
  within ``window`` epochs, and this condition persists for at least
  ``sustain_epochs`` consecutive epochs, mark the run unstable.

All public functions share their core logic via the private
``_scan_instability`` helper to avoid code duplication.
"""

from __future__ import annotations


def _scan_instability(
    val_accuracies: list[float],
    *,
    drop_threshold: float,
    window: int,
    burn_in: int,
    sustain_epochs: int,
) -> int | None:
    """
    Scan ``val_accuracies`` and return the epoch index of first detected
    instability, or ``None`` if no instability is found.

    This is the single implementation shared by ``is_unstable`` and
    ``get_instability_epoch`` — keep them in sync by editing here only.
    """
    if len(val_accuracies) <= burn_in:
        return None

    running_peak_val = val_accuracies[0]
    running_peak_idx = 0
    consecutive_violations = 0

    for i, acc in enumerate(val_accuracies):
        # Update peak — use latest occurrence to anchor window to recent peak.
        if acc >= running_peak_val:
            running_peak_val = acc
            running_peak_idx = i

        if i <= burn_in:
            continue

        drop = running_peak_val - acc
        epochs_since_peak = i - running_peak_idx
        violating = drop > drop_threshold and epochs_since_peak <= window

        if violating:
            consecutive_violations += 1
            if consecutive_violations >= sustain_epochs:
                return i
        else:
            consecutive_violations = 0

    return None


def is_unstable(
    val_accuracies: list[float],
    *,
    drop_threshold: float = 0.08,
    window: int = 5,
    burn_in: int = 10,
    sustain_epochs: int = 1,
) -> bool:
    """
    Return ``True`` if the run exhibits instability / degeneration.

    Parameters
    ----------
    val_accuracies : list[float]
        Validation accuracy per epoch in [0, 1].
    drop_threshold : float
        Minimum drop below running peak to count as a violation.
    window : int
        Maximum epochs since most recent peak to qualify the drop.
    burn_in : int
        Ignore early epochs before this index.
    sustain_epochs : int
        Number of consecutive violating epochs required for detection.
    """
    if sustain_epochs < 1:
        raise ValueError(f"sustain_epochs must be >= 1, got {sustain_epochs}")
    return _scan_instability(
        val_accuracies,
        drop_threshold=drop_threshold,
        window=window,
        burn_in=burn_in,
        sustain_epochs=sustain_epochs,
    ) is not None


def get_instability_epoch(
    val_accuracies: list[float],
    *,
    drop_threshold: float = 0.08,
    window: int = 5,
    burn_in: int = 10,
    sustain_epochs: int = 1,
) -> int | None:
    """
    Return epoch index where instability is first detected, or ``None``.
    """
    if sustain_epochs < 1:
        raise ValueError(f"sustain_epochs must be >= 1, got {sustain_epochs}")
    return _scan_instability(
        val_accuracies,
        drop_threshold=drop_threshold,
        window=window,
        burn_in=burn_in,
        sustain_epochs=sustain_epochs,
    )


def label_run(
    val_accuracies: list[float],
    **kwargs,
) -> dict[str, bool | int | None]:
    """
    Convenience wrapper returning both binary label and first detection epoch.
    """
    epoch = get_instability_epoch(val_accuracies, **kwargs)
    return {"unstable": epoch is not None, "instability_epoch": epoch}
