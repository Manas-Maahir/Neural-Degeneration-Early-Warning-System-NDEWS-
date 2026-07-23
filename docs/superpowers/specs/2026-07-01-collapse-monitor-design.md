# Design: `ndews` — Model-Agnostic Collapse Monitor

**Date:** 2026-07-01
**Status:** Approved (design phase)
**Author:** Manas Maahir (with Claude Code)

## Motivation

The existing codebase is a well-engineered CIFAR-10 research harness for detecting
training instability via 6 internal signals (effective rank, feature reuse, gradient
diversity, neuron sparsity, representational isotropy, activation scale). Its
engineering quality is high, but it is **not** currently something you can drop into
another project:

- The signal hooks (`SignalLogger`) are already model-agnostic, but everything around
  them (dataset, models, regimes, the CLI) is CIFAR scaffolding.
- The trained `RandomForest` predictor is **not transferable** — its features are
  absolute-magnitude and architecture-specific, so a predictor trained on
  SimpleCNN/CIFAR gives meaningless probabilities on a different architecture/task.
- The live-prediction logic is embedded inside `experiments/baseline__run.py`; there
  is no reusable monitor object.

This project extracts a clean, model-agnostic **`CollapseMonitor`** that any PyTorch
training loop can use, with an alert engine that works on day one with **no
pretraining**.

## Goals

- Ship a pip-installable `ndews` package exposing `CollapseMonitor`.
- Work on **any** `torch.nn.Module` and **any** dataset — zero CIFAR assumptions in
  the core library.
- Fire meaningful alerts with **no pretrained model and no user-collected data**, via
  self-baselined anomaly detection over the 6 signals.
- Keep the existing supervised predictor and val-accuracy labeller as **optional**
  layers on top.
- Preserve the existing CIFAR research harness as `examples/` that consume the library
  (proof-of-use + validation harness).

## Non-Goals (v1 / YAGNI)

- No CLI, no dashboard/TUI.
- No PyTorch Lightning / HuggingFace framework callbacks.
- No per-step (sub-epoch) monitoring — epoch granularity only.
- No auto-training of the predictor.

All deferrable to a future version.

## Design Decisions (resolved during brainstorming)

1. **Tool shape:** model-agnostic telemetry monitor (not a shipped pretrained
   classifier, not a paper-repo).
2. **Integration surface:** manual 3-line hook — construct `CollapseMonitor`, call
   `on_epoch_end(val_acc)` each epoch, read the returned report.
3. **Default alert engine:** self-baselined anomaly detection (per-signal z-score vs
   the run's own rolling baseline). The RF predictor and the labeller are optional.
4. **v1 scope:** library + pip package; CIFAR code becomes `examples/`.
5. **Package restructure:** rename `src/` → `ndews/` (correct import path), not a
   thin façade.
6. **Train/eval separation:** `train_only` hook guard — hooks record only when the
   module is in `training` mode, so validation forward passes don't pollute the
   epoch's signal buffers. Order-independent, preserves the 3-line API.

## Package Structure

```
ndews/                      # renamed from src/  (import ndews)
├── __init__.py             # public API: CollapseMonitor, MonitorReport, suggest_layers, __version__
├── signals.py              # SignalLogger + 6 hooks  (+ train_only guard)
├── anomaly.py              # NEW — RollingBaseline, directional z-score alert engine
├── monitor.py              # NEW — CollapseMonitor, MonitorReport
├── labeller.py             # unchanged (optional collapse confirmation)
├── predictor.py            # unchanged (optional supervised layer)
├── evaluation.py           # unchanged (research/validation)
├── seed_utils.py           # unchanged
└── py.typed                # NEW — ship type hints
examples/                   # was the CIFAR harness; now USES the library
├── dataset.py  model.py  regimes.py  train.py   # moved from src/
├── run_pipeline.py  baseline_run.py  run_many_regimes.py
└── quickstart.py           # NEW — minimal drop-in demo
analysis/                   # unchanged (signal_lag, plots)
tests/                      # NEW — real tests for the public API
```

The core `ndews` package carries no dataset/architecture assumptions. All CIFAR-specific
code moves to `examples/` and imports `ndews`.

## Public API

```python
from ndews import CollapseMonitor

monitor = CollapseMonitor(
    model,
    layers=["layer3", "fc"],      # named modules to hook; None -> suggest_layers(model)
    baseline_epochs=5,            # warmup epochs before anomaly scoring begins
    z_threshold=2.5,              # per-signal drift cutoff (in sigma)
    min_signals=2,                # how many signals must drift to raise an alert
    predictor=None,               # optional: path str or Predictor for supervised proba
)

for epoch in range(epochs):
    train_one_epoch(...)          # model.train() -> hooks record
    acc = validate(...)           # model.eval()  -> hooks skip (train_only guard)
    report = monitor.on_epoch_end(val_acc=acc)   # val_acc is optional
    if report.alert:
        print(report.status, report.drifting_signals, report.probability)

monitor.close()                   # or use `with CollapseMonitor(...) as monitor:`
```

**Lifecycle (internal):**
- `__init__` registers hooks and primes the buffers.
- `on_epoch_end` reads the accumulated (train-only) signals, aggregates them, scores
  them against the rolling baseline, builds a `MonitorReport`, then resets the hook
  buffers for the next epoch. No `on_epoch_start` call is required.
- `layers=None` → `suggest_layers(model)` picks the last conv-like module and the last
  linear module, printing a notice (never silent).
- `close()` removes all hooks; context-manager support wraps this.

## `MonitorReport`

Frozen dataclass returned by `on_epoch_end`:

| field | type | meaning |
|---|---|---|
| `epoch` | int | 1-based epoch index |
| `signals` | dict[str, float] | raw per-layer signal dict (full detail) |
| `aggregated` | dict[str, float] | 6 canonical signals (via `canonical_aggregate_features`) |
| `z_scores` | dict[str, float] | per-signal directional z vs rolling baseline |
| `drifting_signals` | list[str] | signals past `z_threshold` in the collapse-ward direction |
| `alert` | bool | `True` when `len(drifting_signals) >= min_signals` |
| `status` | str | `"warming_up"` / `"ok"` / `"warning"` / `"collapse"` |
| `probability` | float \| None | optional supervised collapse proba, else `None` |
| `collapse_flag` | bool \| None | optional labeller confirmation (needs `val_acc` history), else `None` |
| `message` | str | one-line human-readable summary |

## Anomaly Engine (`anomaly.py`)

- `RollingBaseline` collects the first `baseline_epochs` values per signal as a
  mean/std baseline. During warmup, `status="warming_up"` and no alerts are raised.
- After warmup, each signal receives a **directional** z-score using a sign map that
  encodes which direction indicates collapse:
  - effective rank ↓ (down is bad)
  - representational isotropy ↓
  - gradient diversity ↓
  - feature reuse ↑
  - neuron sparsity ↑
  - activation scale ↑ (magnitude drift toward explosion; extreme values are bad)
  Only collapse-ward drift contributes to `drifting_signals`.
- Alert when `>= min_signals` signals cross `z_threshold`.
- `status` resolution:
  - `"warming_up"` while collecting baseline.
  - `"ok"` when no alert.
  - `"warning"` when the anomaly engine alerts.
  - `"collapse"` when the anomaly engine alerts **and** the labeller (if `val_acc`
    history is available) confirms an instability epoch.
- Pure NumPy, no training, fully architecture-agnostic.

## `signals.py` Change

Add `train_only: bool = True` to `SignalLogger.__init__`. Each hook early-returns when
`not module.training`. This cleanly separates train-time signals from eval-time forward
passes without any API burden. Default-on; existing callers (which already read signals
before eval) are unaffected.

## Error Handling

- **Unknown layer names:** existing `[SignalLogger] WARNING` prints the missing layers;
  `CollapseMonitor` raises `ValueError` if **zero** requested layers resolve.
- **Alert before warmup completes:** `status="warming_up"`, never a spurious alert.
- **Predictor supplied but window not yet full:** `probability=None` (mirrors current
  warm-up behaviour in `baseline__run.py`).
- **Non-finite signals** (diverged run): treated as a max-severity drift, not a crash.
- **Predictor schema mismatch:** existing `Predictor.load` warning path is surfaced;
  the monitor continues with anomaly-only alerts.

## Testing (`tests/`, pytest)

Real coverage for the new public surface, all on a tiny `TestMLP` (no CIFAR download):

- Monitor lifecycle and hook cleanup (`close`, context manager).
- Train-only guard: eval-mode forward passes do not record signals.
- Warmup → alert transition driven by synthetic drifting signals.
- `min_signals` / `z_threshold` behaviour.
- `MonitorReport` field population (including optional predictor + labeller paths).
- `suggest_layers` on a couple of architectures.
- First-ever test of `create_sliding_windows` labelling (`forecast` and `detect`
  modes), since correctness of the optional predictor path depends on it.

## Packaging

- `pyproject.toml`: `[tool.setuptools.packages.find]` includes `ndews*` only (drop
  `src*`; `analysis*` optional). Add `py.typed` to package data.
- README quickstart section showing the 3-line integration.
- Version stays `0.1.0` (or bump to `0.2.0` given the API introduction — decide at
  implementation time).

## Rollout / Migration

- All internal imports `from src.x` → `from ndews.x` (one mechanical pass).
- `experiments/` and `run_pipeline.py` move under `examples/` and update imports.
- `analysis/` imports updated (`from src.` → `from ndews.`).
- `test_signals.py` moves into `tests/` and updates imports.
- The existing single-file Colab notebook build (`build_colab_notebook.py`) updated to
  the new module paths if retained.

## Open Questions — resolved during implementation

- **`suggest_layers` heuristic — resolved.** Picks the *last* conv-like module
  (`nn.Conv1d/2d/3d`) and the *last* `nn.Linear`, via `isinstance` in
  [ndews/monitor.py](../../../ndews/monitor.py) (`suggest_layers`). Prints its choice; an
  MLP yields just its last linear.
- **`activation_scale` two-sided — resolved: yes, two-sided (`0`).** Confirmed empirically
  against 25 recorded runs (baseline = each run's first 5 epochs; `*_activation_scale`
  deviation in the collapse region vs that baseline). Collapse drives the signal mostly
  **up/explosion** (`label_noise` ≈ +8.3z, `over_regularization` ≈ +3.6z, exploding
  `delayed_collapse` layers) but genuinely **down/vanishing** in a meaningful minority —
  most clearly within `delayed_collapse`, where one layer explodes while another vanishes
  (≈4 up / 4 down across its runs). A one-sided ↑ rule would miss the vanishing layers, so
  `COLLAPSE_DIRECTIONS["activation_scale"] = 0` (two-sided) stands in
  [ndews/anomaly.py](../../../ndews/anomaly.py). Caveat: `high_learning_rate` reads as
  "no drift" only because it collapses *inside* the 5-epoch baseline window (contaminated
  baseline), not a counter-example. The `min_signals ≥ 2` + `z_threshold` gate guards
  against a lone two-sided signal firing on benign fluctuation.
