"""
scripts/evaluate_predictor.py
==============================
Load a completed batch experiment session and run leave-one-run-out
cross-validation (LORO-CV) against heuristic baselines.

Usage
-----
    python scripts/evaluate_predictor.py \\
        --session-dir output/instability_runs/session_<stamp>

    # With plots saved to session dir:
    python scripts/evaluate_predictor.py \\
        --session-dir output/instability_runs/session_<stamp> \\
        --save-plots

    # Tune window / horizon:
    python scripts/evaluate_predictor.py \\
        --session-dir output/instability_runs/session_<stamp> \\
        --window-size 5 --forecast-horizon 3

Input
-----
Reads all CSV files from ``<session-dir>/runs/*.csv``.
Each CSV must be a per-epoch log produced by ``examples/run_many_regimes.py``
with columns: run_id, epoch, val_acc, instability_epoch, and signal columns
ending in CANONICAL_METRIC_SUFFIXES.

Output
------
- LORO-CV per-fold metric table printed to stdout
- Aggregate mean ± std across folds
- Comparison table: RF predictor vs. MajorityClass, ValAccDrop, Random baselines
- Optional plots saved to ``<output-dir>/``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as `python scripts/evaluate_predictor.py` without installing.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ndews.evaluation import (
    RunData,
    evaluate_baselines,
    leave_one_run_out_cv,
    print_comparison_table,
)
from ndews.signals import CANONICAL_METRIC_SUFFIXES


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _is_signal_column(col: str) -> bool:
    return any(col.endswith(suffix) for suffix in CANONICAL_METRIC_SUFFIXES)


def load_runs_from_session(session_dir: Path) -> list[RunData]:
    """
    Load all per-run CSVs from ``<session-dir>/runs/`` into RunData objects.

    Returns only runs with at least one epoch row. Runs with CSV errors are
    skipped with a warning.
    """
    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"No 'runs/' subdirectory found under {session_dir}. "
            "Make sure you pass a session directory produced by run_many_regimes.py."
        )

    csv_files = sorted(runs_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {runs_dir}")

    runs: list[RunData] = []
    signal_cols: list[str] = []

    for csv_path in csv_files:
        try:
            rows = _load_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {csv_path.name}: {exc}")
            continue

        if not rows:
            continue

        # Discover signal columns from first file that has rows.
        if not signal_cols:
            signal_cols = [c for c in rows[0].keys() if _is_signal_column(c)]

        # Sort by epoch.
        try:
            rows.sort(key=lambda r: int(r.get("epoch", 0)))
        except ValueError:
            pass

        last = rows[-1]
        run_id = last.get("run_id", csv_path.stem)

        # Instability epoch: last non-empty value in the run's rows.
        instability_epoch: int | None = None
        for row in reversed(rows):
            raw = row.get("instability_epoch", "")
            if raw not in ("", "None", None):
                try:
                    instability_epoch = int(float(raw))
                    break
                except ValueError:
                    pass

        # Build per-epoch signal dicts.
        metrics_sequence: list[dict[str, float]] = []
        val_accuracies: list[float] = []

        for row in rows:
            sig_dict = {
                col: float(row[col])
                for col in signal_cols
                if row.get(col, "") not in ("", "None", None)
                and _parse_float(row.get(col, "")) is not None
            }
            metrics_sequence.append(sig_dict)

            val_raw = row.get("val_acc", "")
            val = _parse_float(val_raw)
            val_accuracies.append(val if val is not None else 0.0)

        if not metrics_sequence:
            print(f"[WARN] {csv_path.name}: no usable epoch rows — skipping")
            continue

        runs.append(RunData(
            run_id=run_id,
            metrics_sequence=metrics_sequence,
            val_accuracies=val_accuracies,
            instability_epoch=instability_epoch,
        ))

    print(
        f"[INFO] Loaded {len(runs)} runs from {runs_dir} "
        f"({sum(1 for r in runs if r.instability_epoch is not None)} unstable, "
        f"{sum(1 for r in runs if r.instability_epoch is None)} stable)"
    )
    return runs


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LORO-CV evaluation of the instability predictor vs. baselines."
    )
    parser.add_argument(
        "--session-dir", required=True, type=str,
        help="Path to a session directory produced by run_many_regimes.py.",
    )
    parser.add_argument("--window-size",      type=int,   default=3)
    parser.add_argument("--forecast-horizon", type=int,   default=2)
    parser.add_argument(
        "--val-acc-drop-threshold", type=float, default=0.05,
        help="Drop threshold for ValAccDropBaseline (default 0.05).",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save CV metric and feature importance plots.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save plots (default: <session-dir>/eval/).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir)

    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else session_dir / "eval"

    print("=" * 80)
    print("NDEWS — LORO-CV Predictor Evaluation")
    print(f"session_dir      = {session_dir}")
    print(f"window_size      = {args.window_size}")
    print(f"forecast_horizon = {args.forecast_horizon}")
    print("=" * 80)

    runs = load_runs_from_session(session_dir)

    if len(runs) < 2:
        print(
            f"[ERROR] Need at least 2 runs for LORO-CV, got {len(runs)}. "
            "Run more experiments with run_many_regimes.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # LORO-CV
    # ------------------------------------------------------------------
    print("\n--- Leave-One-Run-Out Cross-Validation ---")
    cv_result = leave_one_run_out_cv(
        runs,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\n--- Baseline Comparisons ---")
    baseline_results = evaluate_baselines(
        runs,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        val_acc_drop_threshold=args.val_acc_drop_threshold,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print_comparison_table(cv_result, baseline_results)

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if args.save_plots:
        try:
            from analysis.plots import plot_cv_metrics, save_figure

            fig_cv = plot_cv_metrics(cv_result, title="LORO-CV Fold Metrics")
            path_cv = save_figure(fig_cv, output_dir / "cv_metrics.png")
            print(f"[Plot] CV metrics → {path_cv}")

            # Feature importance requires training a full predictor on all data.
            from ndews.predictor import Predictor, canonical_aggregate_features, create_sliding_windows
            from analysis.plots import plot_feature_importance

            agg_seqs = [
                [canonical_aggregate_features(ep) for ep in run.metrics_sequence]
                for run in runs
            ]
            all_keys: set[str] = set()
            for seq in agg_seqs:
                for ep in seq:
                    all_keys.update(ep.keys())
            feature_keys = sorted(all_keys)

            X_all, y_all = [], []
            for run, agg_seq in zip(runs, agg_seqs):
                X_i, y_i = create_sliding_windows(
                    agg_seq,
                    window_size=args.window_size,
                    forecast_horizon=args.forecast_horizon,
                    instability_epoch=run.instability_epoch,
                    feature_keys=feature_keys,
                    label_mode="detect",
                )
                X_all.extend(X_i)
                y_all.extend(y_i)

            if X_all:
                full_predictor = Predictor(
                    window_size=args.window_size,
                    forecast_horizon=args.forecast_horizon,
                    feature_keys=feature_keys,
                )
                full_predictor.train(X_all, y_all)
                fig_fi = plot_feature_importance(full_predictor, top_n=20)
                path_fi = save_figure(fig_fi, output_dir / "feature_importance.png")
                print(f"[Plot] Feature importance → {path_fi}")

        except ImportError as exc:
            print(f"[WARN] Could not import plot dependencies: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
