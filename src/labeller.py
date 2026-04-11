"""
src/labeller.py
===============
Objective instability detection for training runs.

Default criterion (v2):
- Start checking after ``burn_in`` epochs.
- Track running peak validation accuracy.
- If accuracy drops by more than ``drop_threshold`` from the latest peak
  within ``window`` epochs and this condition persists for
  ``sustain_epochs`` consecutive epochs, mark the run unstable.
"""

from __future__ import annotations


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
    if len(val_accuracies) <= burn_in:
        return False
    if sustain_epochs < 1:
        raise ValueError(f"sustain_epochs must be >= 1, got {sustain_epochs}")

    running_peak_val = val_accuracies[0]
    running_peak_idx = 0
    consecutive_violations = 0

    for i, acc in enumerate(val_accuracies):
        # Peak update uses latest occurrence to anchor window to recent peak.
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
                return True
        else:
            consecutive_violations = 0

    return False


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
    if len(val_accuracies) <= burn_in:
        return None
    if sustain_epochs < 1:
        raise ValueError(f"sustain_epochs must be >= 1, got {sustain_epochs}")

    running_peak_val = val_accuracies[0]
    running_peak_idx = 0
    consecutive_violations = 0

    for i, acc in enumerate(val_accuracies):
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


def label_run(
    val_accuracies: list[float],
    **kwargs,
) -> dict[str, bool | int | None]:
    """
    Convenience wrapper with both binary label and first detection epoch.
    """
    unstable = is_unstable(val_accuracies, **kwargs)
    epoch = get_instability_epoch(val_accuracies, **kwargs) if unstable else None
    return {"unstable": unstable, "instability_epoch": epoch}
