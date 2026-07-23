# Neural Degeneration Early Warning System (NDEWS)

> **Predicting representational collapse and training instability through internal neural telemetry.**

NDEWS is a research framework that identifies "pre-symptomatic" indicators of training failure.
By monitoring internal activations and gradients throughout training, it learns to forecast
catastrophic degradation *before* it appears in validation metrics.

---

## Objective

- **Drop-in, model-agnostic.** `CollapseMonitor` attaches to any `torch.nn.Module` in three lines.
- **Works on day one.** Self-baselined anomaly detection over 6 internal signals — no pretraining
  and no user-collected data required.
- **Optional supervised layer.** A Random Forest forecaster and a val-accuracy labeller layer on
  top when you have historical runs to learn from.

### Scientific scope

Focuses on **training instability** and **representational degeneration** — the physical process
where a model's internal representations lose diversity, isotropy, or scale stability.

- **Includes:** Internal activation monitoring, gradient diversity analysis, early warning prediction.
- **Excludes:** Recursive generated-data collapse (LLM-style "model collapse").

---

## Internal Signals

Six canonical metrics serve as leading indicators of training health. They often shift **before**
validation accuracy drops, giving the predictor time to raise an alert.

| Metric | Suffix | Rationale |
| :----- | :----- | :-------- |
| **Representation Entropy** | `_representation_entropy` | Effective rank of the activation matrix: `exp(H(σ/Σσ))`, range `[1, min(B,D)]`. Close to 1 → representations collapsed to ~1 dimension. |
| **Feature Reuse** | `_feature_reuse` | Mean off-diagonal cosine similarity of the feature Gram matrix. High = redundant, non-diverse feature set. |
| **Gradient Diversity** | `_gradient_diversity` | Variance of output gradients across the batch. Low = monolithic, uninformative training signal. |
| **Neuron Sparsity** | `_neuron_sparsity` | Fraction of near-zero activations. Uses an *adaptive* threshold (1% of mean absolute activation) so the metric is scale-invariant. |
| **Representational Isotropy** | `_representational_isotropy` | `mean_dim_variance / max_dim_variance` — ratio in [0,1]. Approaches 0 when variance concentrates in a few dimensions (dimensional collapse). |
| **Activation Scale** | `_activation_scale` | RMS activation magnitude. Tracks explosion or vanishing across layers. |

---

## Library quickstart (`CollapseMonitor`)

Install the package, then wrap any PyTorch training loop:

```powershell
pip install -e .
```

```python
from ndews import CollapseMonitor

monitor = CollapseMonitor(model, layers=["layer3", "fc"])  # layers=None -> auto-pick
for epoch in range(epochs):
    train_one_epoch(...)                     # model.train() -> hooks record
    acc = validate(...)                      # model.eval()  -> hooks skip (train-only)
    report = monitor.on_epoch_end(val_acc=acc)
    if report.alert:
        print(report.status, report.drifting_signals, report.probability)
monitor.close()                              # or `with CollapseMonitor(...) as monitor:`
```

`report.status` moves `warming_up → ok → warning → collapse`. Alerts come from
self-baselined anomaly detection — per-signal **directional** z-scores against the run's
own rolling baseline — so nothing needs to be pretrained. Passing `predictor=<path>` adds
an optional supervised `probability`; supplying `val_acc` enables labeller confirmation
(the `collapse` status). Run the self-contained demo (no dataset download):

```powershell
python examples\quickstart.py
```

---

## Setup

```powershell
python -m venv instability_env
.\instability_env\Scripts\activate
pip install -r requirements.txt
```

Or install as an editable package (enables `import ndews` from anywhere):

```powershell
pip install -e .
```

---

## Research harness (CIFAR examples)

The `examples/` directory is the original CIFAR-10 study, now a consumer of the `ndews`
library. It validates the signals on real training dynamics.

### 1. Validate the library

```powershell
python -m pytest
```

The suite covers the monitor, anomaly engine, hook train-only guard, sliding-window
labelling, effective-rank/isotropy bounds, canonical key names, and labeller correctness.

### 2. End-to-end smoke test (2 runs, 10 epochs)

```powershell
python examples\run_pipeline.py
```

Runs one healthy and one noisy training regime, saves `output/metrics_log.csv`, trains a
predictor, and prints in-sample evaluation. Completes in a few minutes on CPU.

### 3. Live collapse monitoring

```powershell
# Train and save predictor artifact first:
python examples\run_pipeline.py

# Run a monitored experiment with per-epoch collapse probability:
python examples\baseline_run.py `
    --regime high_learning_rate --epochs 20 `
    --predictor-path .\output\predictor\random_forest.pkl
```

Per epoch you will see:

```
Collapse Probability = 0.73
Effective Rank       = 4.821
Gradient Diversity   = 0.000042
Feature Reuse        = 0.18421
Neuron Sparsity      = 0.4312
Repr. Isotropy       = 0.2104
Activation Scale     = 3.1024
Validation Accuracy  = 38.50%
...
```

---

## Held-Out Evaluation (Research Workflow)

The smoke test trains and evaluates on the *same* two runs — that is not a valid result.
For held-out evaluation, use leave-one-run-out cross-validation (LORO-CV):

```powershell
# Step 1: Generate batch data (5 seeds × 9 regimes × 30 epochs, ~45 runs)
python examples\run_many_regimes.py --runs-per-regime 5 --epochs 30

# Step 2: Run LORO-CV evaluation (RF predictor vs. three baselines)
python scripts\evaluate_predictor.py `
    --session-dir output\instability_runs\session_<stamp>

# Step 3: Signal lead-time analysis (how far ahead do signals deviate?)
python -m analysis.signal_lag `
    --session-dir output\instability_runs\session_<stamp>
```

The comparison table shows the RF predictor alongside:
- `MajorityClassBaseline` — always predicts the majority class
- `ValAccDropBaseline` — alerts when val accuracy drops within a window (no signal features)
- `RandomBaseline` — random at the training-set positive rate

The predictor has research value only if it beats `ValAccDropBaseline` on F1 and ROC-AUC.

---

## Experiment Regimes

Defined in [`examples/regimes.py`](examples/regimes.py):

| Regime | Perturbation |
| :----- | :----------- |
| `normal` | Baseline (lr=1e-3, clean data) |
| `label_noise` | 35% random label corruption |
| `over_regularization` | weight_decay=0.10 |
| `high_learning_rate` | lr=0.05 |
| `overtraining` | 80 epochs, no regularization |
| `class_imbalance` | 5 classes downsampled to 20% |
| `reduced_dataset_size` | 20% of training data |
| `delayed_collapse` | Healthy for 10 epochs, then 50× LR spike at epoch 11 |
| `warm_then_overfit` | 60 epochs, no regularization — learns then memorizes |

Run a single regime:

```powershell
python examples\baseline_run.py --regime delayed_collapse --epochs 30
```

---

## Output Schema

### Per-epoch run CSV (from `run_many_regimes.py`)

Each row is one epoch of one run:

| Column | Description |
| :----- | :---------- |
| `run_id` | Unique run identifier (e.g. `delayed_collapse__seed100`) |
| `regime` | Regime name |
| `seed` | Random seed |
| `epoch` | Epoch number (1-based) |
| `train_loss` | Training loss |
| `val_loss` | Validation loss |
| `val_acc` | Validation accuracy |
| `unstable` | Whether instability was detected at this epoch |
| `instability_epoch` | First epoch where instability was detected |
| `{layer}_representation_entropy` | Effective rank for layer |
| `{layer}_feature_reuse` | Off-diagonal Gram similarity |
| `{layer}_gradient_diversity` | Gradient variance |
| `{layer}_neuron_sparsity` | Fraction near-zero activations |
| `{layer}_representational_isotropy` | Mean/max per-dim variance ratio |
| `{layer}_activation_scale` | RMS activation magnitude |

### Output directory layout

```
output/
├── predictor/
│   └── random_forest.pkl          # Saved predictor (schema version tracked)
├── metrics_log.csv                # Smoke-test pipeline output
└── instability_runs/
    └── session_<stamp>/
        ├── manifest.json          # Run config + totals
        ├── run_summary.csv        # Per-run metadata
        ├── runs/
        │   └── <run_id>.csv       # Per-epoch signals for each run
        └── eval/                  # Created by evaluate_predictor.py
            ├── cv_metrics.png
            └── feature_importance.png
```

---

## Repository Structure

```
ndews/              # the installable library (import ndews) — zero CIFAR assumptions
  monitor.py       # CollapseMonitor + MonitorReport + suggest_layers  (public API)
  anomaly.py       # RollingBaseline + directional z-score alert engine (no pretraining)
  signals.py       # Hook-based signal logger — 6 metrics + train_only guard
  labeller.py      # Instability detection via _scan_instability() (optional)
  predictor.py     # Sliding windows, RandomForest wrapper, schema versioning (optional)
  evaluation.py    # LORO-CV, RunData dataclass, baseline comparisons (research)
  seed_utils.py    # seed_everything() — deterministic across random/numpy/torch/cuda
  py.typed         # ships type hints

examples/           # the CIFAR-10 research harness — now consumes the ndews library
  quickstart.py    # minimal drop-in CollapseMonitor demo (synthetic data, no download)
  regimes.py       # single-source registry for all 9 experiment regimes
  dataset.py       # CIFAR-10 loaders with augmentation + perturbation controls
  model.py         # SimpleCNN, DeepCNN, TestMLP (all accept num_classes)
  train.py         # train_epoch / eval_epoch — NaN-guarded loss
  run_pipeline.py  # end-to-end smoke test (2 runs, 10 epochs, in-sample evaluation)
  baseline_run.py  # single-regime monitored run with live collapse probability
  run_many_regimes.py   # batch runner across all regimes and seeds

analysis/
  signal_lag.py    # temporal precedence analysis; CLI via python -m analysis.signal_lag
  plots.py         # matplotlib figures for CV metrics, signal trajectories, lead times

scripts/
  evaluate_predictor.py # load session CSVs → LORO-CV → baseline comparison table

tests/              # pytest suite for the ndews public surface (no CIFAR download)
```

---

## Status

Active research. Signal infrastructure is complete and theoretically grounded. Evaluation framework (LORO-CV + baselines + temporal precedence analysis) is implemented. Awaiting held-out experiment results to validate predictive lead-time.
