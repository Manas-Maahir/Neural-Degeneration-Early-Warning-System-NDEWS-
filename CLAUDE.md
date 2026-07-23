# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Degeneration Early Warning System (NDEWS) — a research framework for predicting training instability and representational collapse in neural networks. It extracts 6 internal signals (entropy, feature reuse, gradient diversity, sparsity, embedding variance, activation variance) from model layers via PyTorch hooks, trains a Random Forest predictor on sliding windows of those signals, and forecasts instability before validation loss crashes.

## Setup & Commands

```powershell
python -m venv instability_env
.\instability_env\Scripts\activate
pip install -r requirements.txt
```

| Task | Command |
|------|---------|
| Validate hook infrastructure | `python test_signals.py` |
| Full pipeline (generate data → train predictor) | `python examples\run_pipeline.py` |
| Single experiment | `python examples\baseline_run.py --regime normal --epochs 20` |
| Single experiment with live collapse prediction | `python examples\baseline_run.py --regime high_learning_rate --epochs 20 --predictor-path .\output\predictor\random_forest.pkl` |
| Batch experiments across all regimes | `python examples\run_many_regimes.py --runs-per-regime 3 --epochs 20` |
| Signal lead-time analysis (after batch run) | `python -m analysis.signal_lag --session-dir output\instability_runs\session_<stamp>` |

## Architecture

### Data Flow (examples/run_pipeline.py)

1. **Data generation** — Run healthy (`label_noise=0.0`) and unstable (`label_noise=0.8`) training regimes on CIFAR-10
2. **Signal logging** — PyTorch hooks extract 6 metrics per forward/backward pass, averaged at epoch end
3. **CSV export** — Per-epoch signals saved to `metrics_log.csv` (schema: `epoch`, `val_accuracy`, `regime`, `{layer}_{metric}` columns)
4. **Instability labelling** — Validate accuracy histories scanned for sharp drops via `ndews/labeller.py`
5. **Sliding window transform** — Epoch sequences converted to supervised windows (`window_size=3`, `forecast_horizon=2`)
6. **Random Forest training** — 300-tree ensemble saved to `output/predictor/random_forest.pkl`
7. **Online prediction** — During live training, collapse probability emitted per epoch

### Key Modules

**[ndews/signals.py](ndews/signals.py)** — Core instrumentation. `SignalLogger` registers all 6 hooks on named layers (e.g., `["conv2", "fc1"]`). Call `reset()` at epoch start, `get_epoch_signals()` at epoch end for a flat dict of averaged metrics. All hooks use `register_forward_hook` / `register_full_backward_hook` — the model is never modified.

Six hook types (all canonical names exported as `CANONICAL_METRIC_SUFFIXES`):
- `RepresentationEntropyHook` — Effective rank (Roy & Vetterli 2007): `exp(H(σ/Σσ))`, range `[1, min(B,D)]`; drop signals representational collapse
- `FeatureReuseDetector` — Mean off-diagonal Gram cosine similarity; high = redundant features
- `GradientDiversityTracker` — Gradient variance across batch; low = monolithic training signal
- `NeuronSparsityTracker` — Fraction near-zero activations (adaptive threshold: 1% of mean abs activation)
- `RepresentationalIsotropyTracker` — `mean_dim_var / max_dim_var`; close to 0 = dimensional collapse
- `ActivationScaleTracker` — RMS activation magnitude; tracks explosion/vanishing

**[ndews/predictor.py](ndews/predictor.py)** — `Predictor` class wraps `RandomForestClassifier`. `create_sliding_windows()` converts epoch-wise metric dicts into supervised (X, y) arrays. `canonical_aggregate_features()` averages per-metric-type across all layers into a 6-element feature vector for online inference. Schema version tracked in pickle payload.

**[ndews/labeller.py](ndews/labeller.py)** — `is_unstable()` detects accuracy drops ≥ 8% within a 5-epoch window (burn-in 10 epochs). `get_instability_epoch()` returns the first detection epoch or None. Both share `_scan_instability()` to avoid duplication.

**[ndews/evaluation.py](ndews/evaluation.py)** — Held-out evaluation. `RunData` dataclass holds per-run signals + ground truth. `leave_one_run_out_cv()` trains and evaluates the RF predictor via LORO-CV. `evaluate_baselines()` runs `MajorityClassBaseline`, `ValAccDropBaseline`, `RandomBaseline` on the same splits. `print_comparison_table()` shows predictor vs. baselines side-by-side.

**[examples/regimes.py](examples/regimes.py)** — Single source of truth for experiment configuration. `RegimeConfig` dataclass, `REGIME_REGISTRY` dict, `get_regime_config()`, `build_model()`, `resolve_target_layers()`. All experiment scripts import from here.

**[ndews/seed_utils.py](ndews/seed_utils.py)** — `seed_everything(seed)` sets `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.

**[examples/dataset.py](examples/dataset.py)** — `get_cifar_loaders()` produces CIFAR-10 loaders with optional `label_noise`, `class_imbalance`, `imbalance_classes`, and `train_fraction` stress parameters. Defaults to 0 DataLoader workers on Windows.

**[examples/model.py](examples/model.py)** — Three architectures: `SimpleCNN` (recommended hook layers: `conv2`, `fc1`), `DeepCNN` (hook layers: `conv3`, `fc1`), `TestMLP` (for fast unit tests). All accept `num_classes` parameter.

**[analysis/signal_lag.py](analysis/signal_lag.py)** — Temporal precedence analysis. `compute_signal_lags()` measures how many epochs before detected instability each signal shows a z-score > threshold relative to stable-run baseline. `print_lag_table()` displays results. CLI: `python -m analysis.signal_lag --session-dir <path>`.

**[analysis/plots.py](analysis/plots.py)** — Visualization: `plot_val_accuracy_curves`, `plot_signal_trajectories`, `plot_lead_time_bar`, `plot_cv_metrics`, `plot_feature_importance`. All functions return a `matplotlib.Figure`. Use `save_figure(fig, path)` to persist.

### Experiment Regimes

Defined in [examples/regimes.py](examples/regimes.py):

| Regime | Perturbation |
|--------|-------------|
| `normal` | Baseline (lr=1e-3, clean data) |
| `label_noise` | 35% random label corruption |
| `over_regularization` | weight_decay=0.10 |
| `high_learning_rate` | lr=0.05 |
| `overtraining` | 80 epochs |
| `class_imbalance` | 5 classes downsampled to 20% |
| `reduced_dataset_size` | 20% of training data |
| `delayed_collapse` | Normal for 10 epochs, then 50× LR spike at epoch 11 |
| `warm_then_overfit` | 60 epochs, no regularisation — learns then memorises |

### Output Layout

```
output/
├── predictor/random_forest.pkl     # Trained predictor
└── instability_runs_<session_id>/  # Batch results
    ├── run_summary.csv             # Per-run metadata
    ├── manifest.json
    └── run_*.csv                   # Per-run metric histories
```
