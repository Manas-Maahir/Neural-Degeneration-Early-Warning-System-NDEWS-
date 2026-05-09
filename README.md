# Neural Degeneration Early Warning System (NDEWS)

> **Predicting representational collapse and training instability through internal neural telemetry.**

NDEWS is a research framework designed to identify "pre-symptomatic" indicators of training failure. By monitoring internal activations and gradients throughout the training process, it learns to forecast catastrophic degradation before it manifests in validation loss or accuracy curves.

---

###  Objective
*   **Don't wait for the crash.** Detect representational decay as it happens.
*   **Monitor internal health.** Use PyTorch hooks to extract 6 high-fidelity signal metrics.
*   **Predict the future.** Train lightweight forecasting models (Random Forest) to identify impending instability epochs.

###  Scientific Scope
This project focuses on **training instability** and **representational degeneration** - the physical process where a model's internal representations lose diversity or scale.
*   **Includes:** Internal activation monitoring, early warning systems, gradient diversity analysis.
*   **Excludes:** Recursive generated-data collapse (LLM-style "model collapse").

---

##  Internal Signals & Rationale

We track 6 canonical metrics that serve as leading indicators of training health. These signals often shift **before** validation accuracy drops, providing the data needed for proactive intervention.

| Metric | Rationale (The "Why") |
| :--- | :--- |
| **Representation Entropy** | Measures information density. A sharp drop suggests the model is collapsing into a simplistic, low-dimensional state. |
| **Feature Reuse** | Tracks inter-feature correlation. High redundancy means the model is failing to learn diverse, discriminative features. |
| **Gradient Diversity** | Monitors gradient variance across the batch. Low diversity signals that the training signal is becoming monolithic and unstable. |
| **Neuron Sparsity** | The fraction of "dead" or inactive neurons. Rapid increases in sparsity are physical markers of representational decay. |
| **Embedding Variance** | Statistical stability of pooled outputs. Vanishing or exploding variance is a classic precursor to numerical instability. |
| **Activation Variance** | Global stability metric. Helps identify scale-shifts in layers that lead to saturated or dying activations. |

---

##  Pipeline Walkthrough (Why We Are Doing Each Step)

Our primary entrypoint is `run_pipeline.py`, which showcases the full end-to-end framework. Run the script with:

```powershell
python run_pipeline.py
```

Here is a breakdown of what happens and **why**:

### Step 1: Data Generation & GPU Acceleration
**What happens:** We run training experiments on a healthy dataset and a perturbed dataset (e.g., severe label noise).
**Why:** To train a predictor, we need data representing both "stable" and "unstable" training environments.
**Command:**
```bash
# Handled automatically by the pipeline script
python run_pipeline.py
```

### Step 2: Signal Logging (Internal Telemetry)
**What happens:** PyTorch hooks record 6 specific representations during training.
**Why:** Internal metrics act as leading indicators, shifting under the surface before the final performance crash.
**Verify Metrics:**
```bash
# Check the generated log file after running the pipeline
cat metrics_log.csv
```

### Step 3: Instability Labelling
**What happens:** Validation accuracy histories are processed to find the exact "instability epoch".
**Why:** Provides an objective target (y) for our predictor model.

### Step 4: Time-Series Extraction (Sliding Windows)
**What happens:** Metric sequences are transformed into sliding window features.
**Why:** Essential for building a forecasting model that predicts future instability from a history of signals.

### Step 5: Training the Lightweight Predictor
**What happens:** A Random Forest classifier is trained on the sliding windows.
**Why:** Fast and interpretable early-warning system that identifies impending failure.

### Live Collapse Probability During Training
Once a predictor is trained, the baseline runner can print a live probability score:

```powershell
# 1) Train and save predictor artifact
python run_pipeline.py

# 2) Run monitored training with online collapse probability
python experiments\baseline__run.py --regime high_learning_rate --epochs 20 --predictor-path .\output\predictor\random_forest.pkl
```

Per epoch, you will see:
- `Epoch x/y | ...% done`
- `Collapse Probability = 0.xx`
- The full internal metric block (entropy, gradient diversity, feature reuse, sparsity, variances, losses, and validation accuracy).

---

## Advanced Usage: Individual Experiments

If you want to run specific regimes or custom setups manually:

### Run a single baseline experiment
```bash
python experiments\baseline__run.py --regime normal --epochs 20
```

### Run a stress-test experiment (e.g., high label noise)
```bash
python experiments\baseline__run.py --regime label_noise --epochs 20
```

### Run a batch of experiments across different regimes
```bash
python experiments\run_many_regimes.py --runs-per-regime 3 --epochs 20
```

---

## Output schema (per-epoch run CSV)

The tool saves metrics into `metrics_log.csv`. 
Each row includes:
- Structural keys (`epoch`, `val_accuracy`, `regime`)
- Canonical internal metrics for each tracked layer:
  1. `{layer}_representation_entropy`
  2. `{layer}_feature_reuse`
  3. `{layer}_gradient_diversity`
  4. `{layer}_neuron_sparsity`
  5. `{layer}_embedding_variance`
  6. `{layer}_activation_variance`

## Repository Structure

```text
src/
  dataset.py          # CIFAR loaders + perturbations (noise/imbalance/subsample)
  model.py            # SimpleCNN, DeepCNN, TestMLP
  train.py            # train_epoch / eval_epoch loops
  signals.py          # Hook-based internal signal logger
  labeller.py         # Instability definition and labeling logic
  predictor.py        # Sliding window generation and Random Forest model

run_pipeline.py       # End-to-end execution of the system
test_signals.py       # Smoke tests for hooks + labeller
requirements.txt      # Project dependencies
```

## Setup

From the project root:

```powershell
python -m venv instability_env
.\instability_env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick sanity check

```powershell
python test_signals.py
```
You should see all unit tests pass, validating that PyTorch hooks extract metrics correctly and the labeller recognizes instability patterns. 





## Status: Active Research Phase. Core telemetry is functional; predictive models are being refined.
