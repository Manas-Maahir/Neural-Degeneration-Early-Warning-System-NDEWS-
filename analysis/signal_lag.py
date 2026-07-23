"""
analysis/signal_lag.py
======================
Temporal precedence analysis: measure how many epochs BEFORE detected
instability each internal signal shows statistically significant deviation
from its stable-run baseline.

If signals change BEFORE collapse, the predictor can actually forecast ahead.
If they only change AT or AFTER collapse, the system has no predictive value
beyond detecting an already-happening event.

Usage
-----
As a module (import into notebooks or scripts):

    from analysis.signal_lag import compute_signal_lags, print_lag_table
    results = compute_signal_lags(run_csv_path="output/instability_runs/session_xxx/runs/")
    print_lag_table(results)

As a CLI:

    python -m analysis.signal_lag --session-dir output/instability_runs/session_xxx
    python -m analysis.signal_lag --run-csv path/to/run.csv

Output
------
For each signal metric, returns:
- ``lead_epochs``: median epochs before collapse at which signal crossed
  the deviation threshold (positive = precedes collapse)
- ``detection_rate``: fraction of unstable runs where signal showed
  pre-collapse deviation
- ``stable_mean``, ``stable_std``: distribution in stable runs (for normalisation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Signal metric suffixes from ndews/signals.py (duplicated here to avoid
# importing torch when this module is used as a pure analysis script).
_CANONICAL_SUFFIXES = (
    "_representation_entropy",
    "_feature_reuse",
    "_gradient_diversity",
    "_neuron_sparsity",
    "_representational_isotropy",
    "_activation_scale",
)

_LOOKBACK = 15  # epochs before instability to examine


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Read a CSV into a list of row dicts (values as strings)."""
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _load_runs_from_dir(runs_dir: Path) -> list[dict[str, Any]]:
    """
    Load all per-run CSVs from a session runs/ directory.
    Returns a list of run dicts with keys: run_id, regime, unstable,
    instability_epoch, rows (list of epoch row dicts).
    """
    run_files = sorted(runs_dir.glob("*.csv"))
    if not run_files:
        raise FileNotFoundError(f"No CSV files found in {runs_dir}")

    runs: list[dict[str, Any]] = []
    for csv_path in run_files:
        rows = _load_csv(csv_path)
        if not rows:
            continue

        last = rows[-1]
        inst_raw = last.get("instability_epoch", "")
        instability_epoch = int(float(inst_raw)) if inst_raw not in ("", "None", None) else None
        unstable_raw = last.get("unstable", "False")
        unstable = str(unstable_raw).strip().lower() in ("true", "1", "yes")

        runs.append({
            "run_id":            last.get("run_id", csv_path.stem),
            "regime":            last.get("regime", ""),
            "unstable":          unstable,
            "instability_epoch": instability_epoch,
            "rows":              rows,
        })
    return runs


def _load_single_csv(csv_path: Path) -> list[dict[str, Any]]:
    """
    Load a summary-style CSV where each row is one epoch of one run.
    Expects columns: run_id, epoch, unstable, instability_epoch, plus signal columns.
    """
    rows = _load_csv(csv_path)

    # Group by run_id.
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rid = row.get("run_id", "unknown")
        by_run.setdefault(rid, []).append(row)

    runs: list[dict[str, Any]] = []
    for rid, run_rows in by_run.items():
        run_rows.sort(key=lambda r: int(r.get("epoch", 0)))
        last = run_rows[-1]
        inst_raw = last.get("instability_epoch", "")
        instability_epoch = int(float(inst_raw)) if inst_raw not in ("", "None", None) else None
        unstable_raw = last.get("unstable", "False")
        unstable = str(unstable_raw).strip().lower() in ("true", "1", "yes")

        runs.append({
            "run_id":            rid,
            "regime":            last.get("regime", ""),
            "unstable":          unstable,
            "instability_epoch": instability_epoch,
            "rows":              run_rows,
        })
    return runs


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _extract_signal_series(
    rows: list[dict[str, Any]],
    signal_key: str,
) -> list[float | None]:
    """Return the value of ``signal_key`` for each epoch row (None if missing/invalid)."""
    return [_parse_float(row.get(signal_key, "")) for row in rows]


def _discover_signal_keys(runs: list[dict[str, Any]]) -> list[str]:
    """Find all signal columns present in at least one run."""
    keys: set[str] = set()
    for run in runs:
        for row in run["rows"][:1]:  # headers are consistent — check first row
            for k in row.keys():
                if any(k.endswith(suffix) for suffix in _CANONICAL_SUFFIXES):
                    keys.add(k)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_signal_lags(
    *,
    session_dir: str | Path | None = None,
    run_csv: str | Path | None = None,
    lookback: int = _LOOKBACK,
    deviation_threshold_z: float = 2.0,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-signal lead times relative to detected instability.

    Parameters
    ----------
    session_dir : Path | None
        Path to a session directory containing a ``runs/`` subdirectory
        (output of ``run_many_regimes.py``).
    run_csv : Path | None
        Path to a single summary CSV with ``run_id`` column.
        Exactly one of ``session_dir`` or ``run_csv`` must be provided.
    lookback : int
        Number of epochs before instability to examine.
    deviation_threshold_z : float
        Z-score threshold (relative to stable-run mean/std) at which a
        signal is considered "anomalous" in a given epoch.

    Returns
    -------
    dict mapping signal_key → {
        "lead_epochs": float,       median epochs before collapse where anomaly first appears
        "detection_rate": float,    fraction of unstable runs with a pre-collapse anomaly
        "stable_mean": float,
        "stable_std": float,
        "unstable_pre_mean": float, mean over lookback window in unstable runs
    }
    """
    if (session_dir is None) == (run_csv is None):
        raise ValueError("Provide exactly one of session_dir or run_csv.")

    if session_dir is not None:
        runs_dir = Path(session_dir) / "runs"
        runs = _load_runs_from_dir(runs_dir)
    else:
        runs = _load_single_csv(Path(run_csv))  # type: ignore[arg-type]

    signal_keys = _discover_signal_keys(runs)
    if not signal_keys:
        raise ValueError(
            "No canonical signal columns found in the provided data. "
            "Make sure the CSVs were produced by run_many_regimes.py."
        )

    stable_runs   = [r for r in runs if not r["unstable"]]
    unstable_runs = [r for r in runs if r["unstable"] and r["instability_epoch"] is not None]

    if not stable_runs:
        print("[WARNING] No stable runs found — cannot compute baseline distribution.")
    if not unstable_runs:
        print("[WARNING] No unstable runs found — nothing to analyse.")
        return {}

    results: dict[str, dict[str, Any]] = {}

    for sig_key in signal_keys:
        # Stable baseline.
        stable_vals: list[float] = []
        for run in stable_runs:
            series = _extract_signal_series(run["rows"], sig_key)
            stable_vals.extend(v for v in series if v is not None)

        stable_mean = float(np.mean(stable_vals)) if stable_vals else 0.0
        stable_std  = float(np.std(stable_vals))  if stable_vals else 1.0
        if stable_std < 1e-9:
            stable_std = 1.0  # avoid division by zero

        # Per-unstable-run analysis.
        lead_times: list[int] = []
        pre_means:  list[float] = []

        for run in unstable_runs:
            inst_ep = run["instability_epoch"]  # epoch index (0-based)
            rows = run["rows"]
            series = _extract_signal_series(rows, sig_key)

            # Slice the lookback window: [inst_ep - lookback, inst_ep).
            start = max(0, inst_ep - lookback)
            pre_series = [v for v in series[start:inst_ep] if v is not None]

            if not pre_series:
                continue

            pre_means.append(float(np.mean(pre_series)))

            # Find the earliest epoch in the lookback window where the
            # z-score exceeds the threshold.
            anomaly_offset: int | None = None
            for offset, val in enumerate(pre_series):
                z = abs(val - stable_mean) / stable_std
                if z > deviation_threshold_z:
                    anomaly_offset = offset
                    break  # first anomaly in the window

            if anomaly_offset is not None:
                # lead_time = epochs before instability where anomaly appeared
                window_len = len(pre_series)
                lead_times.append(window_len - anomaly_offset)

        results[sig_key] = {
            "lead_epochs":       float(np.median(lead_times)) if lead_times else float("nan"),
            "detection_rate":    len(lead_times) / len(unstable_runs) if unstable_runs else float("nan"),
            "stable_mean":       stable_mean,
            "stable_std":        stable_std,
            "unstable_pre_mean": float(np.mean(pre_means)) if pre_means else float("nan"),
            "n_unstable_runs":   len(unstable_runs),
            "n_stable_runs":     len(stable_runs),
        }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_lag_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted lead-time table sorted by lead_epochs descending."""
    if not results:
        print("[signal_lag] No results to display.")
        return

    # Sort: most predictive (highest lead_epochs & detection_rate) first.
    sorted_keys = sorted(
        results,
        key=lambda k: (
            results[k].get("lead_epochs") or 0.0,
            results[k].get("detection_rate") or 0.0,
        ),
        reverse=True,
    )

    col = 52
    header = (
        f"{'Signal':<{col}}"
        f"{'Lead (epochs)':>14}"
        f"{'Det. Rate':>12}"
        f"{'Stable μ':>12}"
        f"{'Stable σ':>12}"
        f"{'PreCrash μ':>12}"
    )
    print("\n" + "=" * len(header))
    print("Signal Lead-Time Analysis (epochs BEFORE detected collapse)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key in sorted_keys:
        r = results[key]
        lead = r.get("lead_epochs", float("nan"))
        det  = r.get("detection_rate", float("nan"))
        sm   = r.get("stable_mean", float("nan"))
        ss   = r.get("stable_std", float("nan"))
        pm   = r.get("unstable_pre_mean", float("nan"))

        lead_str = f"{lead:.1f}" if not (isinstance(lead, float) and lead != lead) else "—"
        det_str  = f"{det:.2f}"  if not (isinstance(det,  float) and det  != det)  else "—"

        print(
            f"{key:<{col}}"
            f"{lead_str:>14}"
            f"{det_str:>12}"
            f"{sm:>12.4f}"
            f"{ss:>12.4f}"
            f"{pm:>12.4f}"
        )

    print("=" * len(header))
    first = results[sorted_keys[0]]
    print(
        f"\nNote: analysis over {first['n_unstable_runs']} unstable runs, "
        f"{first['n_stable_runs']} stable runs."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute signal lead-times relative to instability onset."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-dir", type=str, default=None,
        help="Path to a session directory (contains runs/ subdir).",
    )
    group.add_argument(
        "--run-csv", type=str, default=None,
        help="Path to a single per-epoch CSV with run_id column.",
    )
    parser.add_argument(
        "--lookback", type=int, default=_LOOKBACK,
        help=f"Epochs before instability to examine (default {_LOOKBACK}).",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=2.0,
        help="Z-score threshold for anomaly detection (default 2.0).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        lag_results = compute_signal_lags(
            session_dir=args.session_dir,
            run_csv=args.run_csv,
            lookback=args.lookback,
            deviation_threshold_z=args.z_threshold,
        )
        print_lag_table(lag_results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
