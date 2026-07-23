"""
build_colab_notebook.py
=======================
Assembles the multi-module NDEWS project into a single Colab-ready notebook
(`Model_Collapse_Prediction.ipynb`).

It reads each library module from `ndews/`, `examples/`, and `analysis/`, strips
intra-project imports (`from ndews...`, `from examples...`, `from analysis...`),
`from __future__` lines, and any `if __name__ == "__main__"` CLI blocks, then emits
each as its own code cell. The CLI scripts (run_pipeline, run_many_regimes,
baseline_run, evaluate_predictor) are NOT pasted verbatim — they are re-expressed as
notebook-native runner cells defined at the bottom of this file.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "Model_Collapse_Prediction.ipynb"

# Modules inlined in dependency order.
MODULE_FILES = [
    "ndews/seed_utils.py",
    "examples/model.py",
    "examples/regimes.py",
    "examples/dataset.py",
    "examples/train.py",
    "ndews/signals.py",
    "ndews/predictor.py",
    "ndews/labeller.py",
    "ndews/evaluation.py",
    "analysis/signal_lag.py",
    "analysis/plots.py",
]

_DROP_PREFIXES = (
    "from __future__ import",
    "from ndews.",
    "from ndews ",
    "import ndews",
    "from examples.",
    "from examples ",
    "import examples",
    "from analysis.",
    "import analysis",
)


def clean_module(path: Path) -> str:
    """Return module source with intra-project imports and CLI block removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    skipping_paren_import = False
    for line in lines:
        stripped = line.strip()
        if skipping_paren_import:
            # Inside a dropped multi-line `from X import (...)` block.
            if ")" in stripped:
                skipping_paren_import = False
            continue
        if stripped.startswith("if __name__"):
            break  # drop CLI runner block and everything after
        if any(stripped.startswith(p) for p in _DROP_PREFIXES):
            if "(" in stripped and ")" not in stripped:
                skipping_paren_import = True  # continuation lines follow
            continue
        if 'matplotlib.use("Agg")' in line:
            continue  # keep inline rendering in the notebook
        out.append(line)
    body = "\n".join(out).strip("\n")
    return "from __future__ import annotations\n\n" + body


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
cells.append(md(
    "# Neural Degeneration Early-Warning System (NDEWS)\n"
    "\n"
    "**Self-contained Colab notebook.** Predicts training instability / representational\n"
    "collapse in CNNs by extracting 6 internal signals via PyTorch hooks, then training a\n"
    "Random-Forest forecaster on sliding windows of those signals.\n"
    "\n"
    "This notebook bundles the entire `ndews/` + `examples/` + `analysis/` codebase into one file.\n"
    "\n"
    "**How to run:** `Runtime -> Run all`. For a GPU: `Runtime -> Change runtime type -> GPU`.\n"
    "\n"
    "Pipeline: install deps -> define library -> quick smoke test -> batch experiments ->\n"
    "leave-one-run-out evaluation vs. baselines -> signal lead-time analysis -> plots.\n"
))

cells.append(md("## 0. Install dependencies"))
cells.append(code(
    "# Colab already ships torch/torchvision/sklearn/matplotlib; install is a no-op there.\n"
    "# Uncomment if running on a bare environment.\n"
    "# !pip install -q torch torchvision scikit-learn matplotlib tqdm numpy\n"
    "import torch, torchvision, sklearn, numpy, matplotlib\n"
    "print('torch', torch.__version__, '| torchvision', torchvision.__version__,\n"
    "      '| sklearn', sklearn.__version__)\n"
))

cells.append(md(
    "## 1. Library code\n"
    "\n"
    "Each cell below is one module from the project, inlined verbatim (intra-project\n"
    "imports and CLI blocks stripped). Run them top to bottom."
))

_TITLES = {
    "ndews/seed_utils.py": "### 1.1 `seed_utils` — deterministic seeding",
    "examples/model.py": "### 1.2 `model` — SimpleCNN / DeepCNN / TestMLP",
    "examples/regimes.py": "### 1.3 `regimes` — experiment configs + model builder",
    "examples/dataset.py": "### 1.4 `dataset` — CIFAR-10 loaders with stress controls",
    "examples/train.py": "### 1.5 `train` — train/eval epoch loops",
    "ndews/signals.py": "### 1.6 `signals` — the 6 hook signals + SignalLogger",
    "ndews/predictor.py": "### 1.7 `predictor` — sliding windows + RandomForest predictor",
    "ndews/labeller.py": "### 1.8 `labeller` — instability detection from val-accuracy",
    "ndews/evaluation.py": "### 1.9 `evaluation` — LORO-CV + heuristic baselines",
    "analysis/signal_lag.py": "### 1.10 `signal_lag` — signal lead-time analysis",
    "analysis/plots.py": "### 1.11 `plots` — visualisation helpers",
}

for rel in MODULE_FILES:
    cells.append(md(_TITLES[rel]))
    cells.append(code(clean_module(ROOT / rel)))

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
cells.append(md("## 2. Runtime configuration"))
cells.append(code(
    "%matplotlib inline\n"
    "import torch\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('Device:', DEVICE)\n"
    "if DEVICE.type == 'cuda':\n"
    "    print('GPU   :', torch.cuda.get_device_name(0))\n"
))

# ---------------------------------------------------------------------------
# Quick smoke test (adapted from run_pipeline.py)
# ---------------------------------------------------------------------------
cells.append(md(
    "## 3. Quick smoke test\n"
    "\n"
    "Trains one healthy run and one run that collapses mid-training (a 50x LR spike at\n"
    "epoch 6) on a 10% CIFAR-10 subsample, then trains a Random-Forest predictor on their\n"
    "signals. The LR spike creates a genuine healthy->collapse transition so both classes are\n"
    "present. **In-sample only** — a fast end-to-end check that the pipeline executes.\n"
    "(Adapted from `run_pipeline.py`.)"
))
cells.append(code(
    "import csv\n"
    "from pathlib import Path\n"
    "import torch.nn as nn\n"
    "\n"
    "def quick_smoke_test(epochs=10, batch_size=128, seed=42,\n"
    "                     window_size=3, forecast_horizon=2,\n"
    "                     output_dir='output'):\n"
    "    seed_everything(seed)\n"
    "    output_dir = Path(output_dir)\n"
    "    output_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "    def run_one(regime_name, label_noise=0.0, lr_spike=None):\n"
    "        # lr_spike=(epoch_index, factor): multiply LR mid-run to force a real\n"
    "        # healthy->collapse transition (a forecastable event, unlike label noise\n"
    "        # which is merely bad from the start).\n"
    "        print(f'\\n=> Regime: {regime_name}  (device={DEVICE})')\n"
    "        train_loader, val_loader = get_cifar_loaders(\n"
    "            batch_size=batch_size, label_noise=label_noise, train_fraction=0.1)\n"
    "        model = SimpleCNN().to(DEVICE)\n"
    "        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
    "        criterion = nn.CrossEntropyLoss()\n"
    "        logger = SignalLogger(model, target_layers=['conv2', 'fc1'])\n"
    "        metrics_history, val_acc_history = [], []\n"
    "        for epoch in range(epochs):\n"
    "            if lr_spike is not None and epoch == lr_spike[0]:\n"
    "                for pg in optimizer.param_groups:\n"
    "                    pg['lr'] *= lr_spike[1]\n"
    "                print(f'  [LR spike] epoch {epoch+1}: lr x{lr_spike[1]}')\n"
    "            logger.reset()\n"
    "            train_loss = train_epoch(model, train_loader, optimizer, criterion,\n"
    "                                     DEVICE, show_progress=False)\n"
    "            signals = logger.get_epoch_signals()\n"
    "            _, val_acc = eval_epoch(model, val_loader, criterion, DEVICE,\n"
    "                                    show_progress=False)\n"
    "            print(f'  Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')\n"
    "            metrics_history.append(signals)\n"
    "            val_acc_history.append(val_acc)\n"
    "        logger.remove_hooks()\n"
    "        return metrics_history, val_acc_history\n"
    "\n"
    "    print('--- Phase 1: Data Generation ---')\n"
    "    healthy_metrics, healthy_acc = run_one('Healthy')\n"
    "    unstable_metrics, unstable_acc = run_one('Unstable', lr_spike=(6, 50.0))\n"
    "\n"
    "    print('\\n--- Phase 2: Instability Labelling ---')\n"
    "    healthy_inst = get_instability_epoch(healthy_acc, drop_threshold=0.02,\n"
    "                                         burn_in=2, chance_level=0.10)\n"
    "    unstable_inst = get_instability_epoch(unstable_acc, drop_threshold=0.02,\n"
    "                                          burn_in=2, chance_level=0.10)\n"
    "    print(f'  Healthy  instability epoch: {healthy_inst}')\n"
    "    print(f'  Unstable instability epoch: {unstable_inst}')\n"
    "\n"
    "    print('\\n--- Phase 3: Train Random Forest Predictor ---')\n"
    "    feat_h = [canonical_aggregate_features(m) for m in healthy_metrics]\n"
    "    feat_u = [canonical_aggregate_features(m) for m in unstable_metrics]\n"
    "    X_h, y_h, feature_keys = create_sliding_windows(\n"
    "        feat_h, window_size=window_size, forecast_horizon=forecast_horizon,\n"
    "        instability_epoch=healthy_inst, label_mode='detect', return_feature_keys=True)\n"
    "    X_u, y_u = create_sliding_windows(\n"
    "        feat_u, window_size=window_size, forecast_horizon=forecast_horizon,\n"
    "        instability_epoch=unstable_inst, feature_keys=feature_keys, label_mode='detect')\n"
    "    X_train, y_train = X_h + X_u, y_h + y_u\n"
    "    if not X_train:\n"
    "        print('Not enough epochs to create sliding windows.'); return None\n"
    "    predictor = Predictor(window_size=window_size, forecast_horizon=forecast_horizon,\n"
    "                          feature_keys=feature_keys)\n"
    "    predictor.train(X_train, y_train)\n"
    "    saved = predictor.save(output_dir / 'predictor' / 'random_forest.pkl')\n"
    "    print(f'[INFO] Predictor saved -> {saved}')\n"
    "    print(f'\\n[EVAL] In-sample evaluation on {len(X_train)} windows (NOT held-out):')\n"
    "    predictor.evaluate(X_train, y_train)\n"
    "    print('\\nSmoke test complete.')\n"
    "    return predictor\n"
    "\n"
    "_ = quick_smoke_test()\n"
))

# ---------------------------------------------------------------------------
# Batch experiments (adapted from run_many_regimes.py)
# ---------------------------------------------------------------------------
cells.append(md(
    "## 4. Batch experiments across regimes\n"
    "\n"
    "Runs each regime across several seeds, logs per-epoch signals to CSV, and returns\n"
    "`RunData` objects in memory for evaluation. Adapted from `experiments/run_many_regimes.py`.\n"
    "\n"
    "Defaults are a Colab-friendly subset. For the full study set\n"
    "`regimes=tuple(ALL_REGIMES)` and more seeds (slower)."
))
cells.append(code(
    "import csv, json\n"
    "from datetime import datetime\n"
    "from pathlib import Path\n"
    "import torch.nn as nn\n"
    "\n"
    "def _canonical_signals(signals):\n"
    "    return {k: v for k, v in signals.items()\n"
    "            if any(k.endswith(s) for s in CANONICAL_METRIC_SUFFIXES)}\n"
    "\n"
    "def _write_csv(path, rows):\n"
    "    if not rows:\n"
    "        return\n"
    "    path.parent.mkdir(parents=True, exist_ok=True)\n"
    "    with path.open('w', newline='', encoding='utf-8') as f:\n"
    "        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n"
    "        w.writeheader(); w.writerows(rows)\n"
    "\n"
    "def run_batch_experiments(\n"
    "    regimes=('normal', 'high_learning_rate', 'delayed_collapse'),\n"
    "    seeds=(100, 101),\n"
    "    epochs_override=None,\n"
    "    model_name='simple',\n"
    "    batch_size=128,\n"
    "    data_root='./data',\n"
    "    output_root='./output/instability_runs',\n"
    "    drop_threshold=0.08, drop_window=5, burn_in=10, sustain_epochs=3,\n"
    "    chance_level=0.10,\n"
    "):\n"
    "    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n"
    "    session_dir = Path(output_root) / f'session_{stamp}'\n"
    "    runs_dir = session_dir / 'runs'\n"
    "    runs_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "    print('=' * 90)\n"
    "    print(f'Batch runner | session={session_dir}')\n"
    "    print(f'regimes={list(regimes)}  seeds={list(seeds)}  '\n"
    "          f'default_model={model_name}  device={DEVICE}')\n"
    "    print('=' * 90)\n"
    "\n"
    "    run_data_list, summaries = [], []\n"
    "    total = len(regimes) * len(seeds)\n"
    "    idx = 0\n"
    "    for regime in regimes:\n"
    "        cfg = get_regime_config(regime)\n"
    "        epochs = epochs_override if epochs_override is not None else cfg.epochs\n"
    "        rmodel = cfg.model or model_name\n"
    "        target_layers = resolve_target_layers(rmodel, None)\n"
    "        for seed in seeds:\n"
    "            idx += 1\n"
    "            run_id = f'{regime}__seed{seed}'\n"
    "            print(f'[{idx:03d}/{total:03d}] {run_id}  epochs={epochs}  lr={cfg.lr:.5f}  '\n"
    "                  f'model={rmodel}  augment={cfg.augment}  frac={cfg.train_fraction}')\n"
    "            seed_everything(seed)\n"
    "            train_loader, val_loader = get_cifar_loaders(\n"
    "                batch_size=batch_size, data_root=data_root,\n"
    "                label_noise=cfg.label_noise, class_imbalance=cfg.class_imbalance,\n"
    "                train_fraction=cfg.train_fraction, augment=cfg.augment, seed=seed)\n"
    "            model = build_model(rmodel).to(DEVICE)\n"
    "            logger = SignalLogger(model, target_layers=target_layers)\n"
    "            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,\n"
    "                                         weight_decay=cfg.weight_decay)\n"
    "            criterion = nn.CrossEntropyLoss()\n"
    "            val_history, rows, metrics_sequence = [], [], []\n"
    "            try:\n"
    "                for epoch in range(1, epochs + 1):\n"
    "                    if cfg.lr_boost_at_epoch is not None and epoch == cfg.lr_boost_at_epoch:\n"
    "                        for pg in optimizer.param_groups:\n"
    "                            pg['lr'] *= cfg.lr_boost_factor\n"
    "                    logger.reset()\n"
    "                    train_loss = train_epoch(model, train_loader, optimizer, criterion,\n"
    "                                             DEVICE, show_progress=False)\n"
    "                    signals = logger.get_epoch_signals()\n"
    "                    val_loss, val_acc = eval_epoch(model, val_loader, criterion,\n"
    "                                                   DEVICE, show_progress=False)\n"
    "                    val_history.append(val_acc)\n"
    "                    collapse = label_run(val_history, drop_threshold=drop_threshold,\n"
    "                                         window=drop_window, burn_in=burn_in,\n"
    "                                         sustain_epochs=sustain_epochs,\n"
    "                                         chance_level=chance_level)\n"
    "                    canon = _canonical_signals(signals)\n"
    "                    metrics_sequence.append(canon)\n"
    "                    row = {'run_id': run_id, 'regime': regime, 'seed': seed,\n"
    "                           'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,\n"
    "                           'val_acc': val_acc, 'peak_val_acc': max(val_history),\n"
    "                           'unstable': bool(collapse['unstable']),\n"
    "                           'instability_epoch': collapse['instability_epoch']}\n"
    "                    row.update(canon)\n"
    "                    rows.append(row)\n"
    "            finally:\n"
    "                logger.remove_hooks()\n"
    "            final = label_run(val_history, drop_threshold=drop_threshold,\n"
    "                              window=drop_window, burn_in=burn_in,\n"
    "                              sustain_epochs=sustain_epochs,\n"
    "                              chance_level=chance_level)\n"
    "            _write_csv(runs_dir / f'{run_id}.csv', rows)\n"
    "            run_data_list.append(RunData(run_id=run_id, metrics_sequence=metrics_sequence,\n"
    "                                         val_accuracies=val_history,\n"
    "                                         instability_epoch=final['instability_epoch']))\n"
    "            summaries.append({'run_id': run_id, 'regime': regime, 'seed': seed,\n"
    "                              'epochs': epochs, 'best_val_acc': max(val_history),\n"
    "                              'final_val_acc': val_history[-1],\n"
    "                              'unstable': bool(final['unstable']),\n"
    "                              'instability_epoch': final['instability_epoch']})\n"
    "            print(f'    done  best={max(val_history)*100:.2f}%  '\n"
    "                  f\"final={val_history[-1]*100:.2f}%  unstable={final['unstable']}\")\n"
    "    _write_csv(session_dir / 'run_summary.csv', summaries)\n"
    "    (session_dir / 'manifest.json').write_text(json.dumps(\n"
    "        {'session_dir': str(session_dir), 'regimes': list(regimes),\n"
    "         'seeds': list(seeds), 'default_model': model_name}, indent=2),\n"
    "        encoding='utf-8')\n"
    "    print('=' * 90)\n"
    "    print(f'Session written to: {session_dir}')\n"
    "    n_unstable = sum(1 for r in run_data_list if r.instability_epoch is not None)\n"
    "    print(f'Runs: {len(run_data_list)}  ({n_unstable} unstable, '\n"
    "          f'{len(run_data_list) - n_unstable} stable)')\n"
    "    return session_dir, run_data_list\n"
))
cells.append(code(
    "# Run the batch. Bump seeds / regimes / epochs for a fuller study.\n"
    "SESSION_DIR, RUNS = run_batch_experiments(\n"
    "    regimes=('normal', 'high_learning_rate', 'delayed_collapse'),\n"
    "    seeds=(100, 101),\n"
    ")\n"
))

# ---------------------------------------------------------------------------
# Evaluation (adapted from scripts/evaluate_predictor.py)
# ---------------------------------------------------------------------------
cells.append(md(
    "## 5. Held-out evaluation (LORO-CV vs. baselines)\n"
    "\n"
    "Leave-one-run-out cross-validation of the RF predictor, compared against majority-class,\n"
    "val-accuracy-drop, and random baselines. Adapted from `scripts/evaluate_predictor.py`.\n"
    "\n"
    "**Framing — early *detection*, not pure forecasting.** A window is positive if the\n"
    "collapse onset falls inside it or within the next `forecast_horizon` epochs\n"
    "(`label_mode='detect'`); pure post-collapse windows are dropped. This is the honest\n"
    "target for these regimes: `delayed_collapse` is triggered by an exogenous LR spike with\n"
    "no internal precursor, so its pre-collapse signals are identical to a healthy run — the\n"
    "model can detect collapse as it begins, not predict an unforeseeable shock. Section 6's\n"
    "lead-time analysis is the complementary 'how early do signals move' view.\n"
    "\n"
    "Each fold also calibrates its decision threshold on the training runs (using the RF's\n"
    "out-of-bag probabilities), so `f1`/`precision`/`recall` reflect a usable alarm\n"
    "operating point — not the saturated 0.5 default. `roc_auc` stays threshold-free."
))
cells.append(code(
    "if len(RUNS) < 2:\n"
    "    print('Need >= 2 runs for LORO-CV. Re-run section 4 with more seeds/regimes.')\n"
    "else:\n"
    "    print('--- Leave-One-Run-Out Cross-Validation ---')\n"
    "    cv_result = leave_one_run_out_cv(RUNS, window_size=3, forecast_horizon=2, verbose=True)\n"
    "    print('\\n--- Baseline Comparisons ---')\n"
    "    baseline_results = evaluate_baselines(RUNS, window_size=3, forecast_horizon=2, verbose=True)\n"
    "    print_comparison_table(cv_result, baseline_results)\n"
))

# ---------------------------------------------------------------------------
# Signal lead-time analysis
# ---------------------------------------------------------------------------
cells.append(md(
    "## 6. Signal lead-time analysis\n"
    "\n"
    "How many epochs *before* detected collapse does each signal deviate from its stable-run\n"
    "baseline? Positive lead = genuine early warning. Reads the CSVs written in section 4."
))
cells.append(code(
    "try:\n"
    "    lag_results = compute_signal_lags(session_dir=SESSION_DIR)\n"
    "    print_lag_table(lag_results)\n"
    "except (FileNotFoundError, ValueError) as exc:\n"
    "    lag_results = {}\n"
    "    print(f'[signal_lag] {exc}')\n"
))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
cells.append(md("## 7. Plots"))
cells.append(code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "# 7.1 Validation-accuracy curves with instability markers.\n"
    "plot_runs = [{'run_id': r.run_id, 'val_accuracies': r.val_accuracies,\n"
    "              'instability_epoch': r.instability_epoch,\n"
    "              'regime': r.run_id.split('__')[0]} for r in RUNS]\n"
    "plot_val_accuracy_curves(plot_runs); plt.show()\n"
))
cells.append(code(
    "# 7.2 LORO-CV per-fold metrics.\n"
    "if 'cv_result' in globals() and cv_result.get('fold_results'):\n"
    "    plot_cv_metrics(cv_result); plt.show()\n"
))
cells.append(code(
    "# 7.3 Signal lead-time bar chart.\n"
    "if lag_results:\n"
    "    plot_lead_time_bar(lag_results); plt.show()\n"
))
cells.append(code(
    "# 7.4 RandomForest feature importance (train one predictor on all runs).\n"
    "agg_seqs = [[canonical_aggregate_features(ep) for ep in r.metrics_sequence] for r in RUNS]\n"
    "all_keys = sorted({k for seq in agg_seqs for ep in seq for k in ep})\n"
    "X_all, y_all = [], []\n"
    "for r, seq in zip(RUNS, agg_seqs):\n"
    "    Xi, yi = create_sliding_windows(seq, window_size=3, forecast_horizon=2,\n"
    "                                    instability_epoch=r.instability_epoch,\n"
    "                                    feature_keys=all_keys, label_mode='detect')\n"
    "    X_all.extend(Xi); y_all.extend(yi)\n"
    "if X_all:\n"
    "    full_predictor = Predictor(window_size=3, forecast_horizon=2, feature_keys=all_keys)\n"
    "    full_predictor.train(X_all, y_all)\n"
    "    plot_feature_importance(full_predictor, top_n=20); plt.show()\n"
    "else:\n"
    "    print('Not enough windows to train a feature-importance model.')\n"
))

# ---------------------------------------------------------------------------
# Section 8 — Genuine forecasting study (endogenous collapse)
# ---------------------------------------------------------------------------
cells.append(md(
    "## 8. Genuine forecasting study (endogenous collapse)\n"
    "\n"
    "Sections 4-7 are *detection*: those regimes (`delayed_collapse`, `high_learning_rate`)\n"
    "collapse from exogenous shocks with no internal precursor, so they can only be caught as\n"
    "they begin. **This section does genuine forecasting** — predicting collapse *before* it\n"
    "happens (`label_mode='forecast'`, positive windows strictly pre-onset).\n"
    "\n"
    "That requires *endogenous* collapse: gradual memorization where the internal signals drift\n"
    "before validation accuracy crashes. We engineer three (augmentation off, tiny/noisy data):\n"
    "`memorization_collapse` (SimpleCNN, tiny clean data), `label_noise_collapse` (fits clean\n"
    "labels then memorizes noise), `deep_memorization` (DeepCNN overfits harder), plus `normal`\n"
    "as a healthy reference.\n"
    "\n"
    "The research question: **how many epochs ahead can internal signals forecast collapse, and\n"
    "do they beat simply watching validation accuracy?** Self-contained — runnable on its own."
))
cells.append(code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "# Endogenous collapses degrade from a peak (not to chance) -> chance_level=None and a\n"
    "# drop detector tuned for gradual degradation. 5 seeds for LORO-CV statistics.\n"
    "FC_SESSION_DIR, FC_RUNS = run_batch_experiments(\n"
    "    regimes=('normal', 'memorization_collapse', 'label_noise_collapse', 'deep_memorization'),\n"
    "    seeds=(100, 101, 102, 103, 104),\n"
    "    chance_level=None,\n"
    "    drop_threshold=0.06, drop_window=8, burn_in=8, sustain_epochs=2,\n"
    ")\n"
))
cells.append(code(
    "# Val-accuracy curves: confirm the engineered runs peak then degrade (a real onset).\n"
    "fc_plot_runs = [{'run_id': r.run_id, 'val_accuracies': r.val_accuracies,\n"
    "                 'instability_epoch': r.instability_epoch,\n"
    "                 'regime': r.run_id.split('__')[0]} for r in FC_RUNS]\n"
    "plot_val_accuracy_curves(fc_plot_runs, title='Endogenous collapse: val accuracy'); plt.show()\n"
))
cells.append(code(
    "# Forecasting skill vs lead time (LORO-CV, forecast mode) vs the val-accuracy baseline.\n"
    "print('--- Forecasting skill vs lead time ---')\n"
    "fc_sweep = forecast_horizon_sweep(FC_RUNS, horizons=(1, 2, 3, 5, 8),\n"
    "                                  window_size=3, verbose=True)\n"
    "plot_horizon_sweep(fc_sweep, metric='roc_auc'); plt.show()\n"
    "plot_horizon_sweep(fc_sweep, metric='f1'); plt.show()\n"
))
cells.append(code(
    "# Which signals move first, and how many epochs before the crash.\n"
    "print('--- Signal lead-time (which signals precede collapse) ---')\n"
    "try:\n"
    "    fc_lag = compute_signal_lags(session_dir=FC_SESSION_DIR)\n"
    "    print_lag_table(fc_lag)\n"
    "    if fc_lag:\n"
    "        plot_lead_time_bar(fc_lag); plt.show()\n"
    "except (FileNotFoundError, ValueError) as exc:\n"
    "    print(f'[signal_lag] {exc}')\n"
))
cells.append(code(
    "# Forecast-mode feature importance: which signals/timesteps drive the forecast.\n"
    "fc_agg = [[canonical_aggregate_features(ep) for ep in r.metrics_sequence] for r in FC_RUNS]\n"
    "fc_keys = sorted({k for seq in fc_agg for ep in seq for k in ep})\n"
    "fc_X, fc_y = [], []\n"
    "for r, seq in zip(FC_RUNS, fc_agg):\n"
    "    Xi, yi = create_sliding_windows(seq, window_size=3, forecast_horizon=3,\n"
    "                                    instability_epoch=r.instability_epoch,\n"
    "                                    feature_keys=fc_keys, label_mode='forecast')\n"
    "    fc_X.extend(Xi); fc_y.extend(yi)\n"
    "if fc_X and sum(fc_y) > 0:\n"
    "    fc_predictor = Predictor(window_size=3, forecast_horizon=3, feature_keys=fc_keys)\n"
    "    fc_predictor.train(fc_X, fc_y)\n"
    "    plot_feature_importance(fc_predictor, top_n=18); plt.show()\n"
    "else:\n"
    "    print('No forecastable positive windows at horizon 3 (see the horizon sweep above).')\n"
))

cells.append(md(
    "---\n"
    "Done. Section 4-7 = detection of exogenous shocks; **section 8 = genuine forecasting of\n"
    "endogenous collapse**. To scale up, add seeds in section 8 or `regimes=tuple(ALL_REGIMES)`\n"
    "in section 4."
))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(cells)} cells)")
