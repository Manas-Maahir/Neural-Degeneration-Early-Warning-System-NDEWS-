"""
examples/quickstart.py
======================
Smallest possible end-to-end CollapseMonitor demo — no CIFAR download, CPU-only.

Trains a tiny classifier on a synthetic dataset, then spikes the learning rate
mid-run to induce instability, so you can watch the monitor move
``warming_up -> ok -> warning/collapse``. The three lines that matter:

    monitor = CollapseMonitor(model)          # attach hooks
    ...
    report = monitor.on_epoch_end(val_acc=acc)  # score each epoch
    ...
    monitor.close()                            # detach hooks

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `ndews` resolves without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ndews import CollapseMonitor


class TinyNet(nn.Module):
    """A minimal MLP — no CIFAR assumptions, works on any [B, in_dim] input."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.head(x)


def make_synthetic_data(n=1024, in_dim=32, num_classes=4, seed=0):
    """A fixed random linear rule makes labels learnable (a genuine train signal)."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(in_dim, num_classes, generator=g)
    x = torch.randn(n, in_dim, generator=g)
    y = (x @ weight).argmax(dim=1)
    return x, y


def _iterate_batches(x, y, batch_size, generator):
    perm = torch.randperm(x.size(0), generator=generator)
    for i in range(0, x.size(0), batch_size):
        idx = perm[i : i + batch_size]
        yield x[idx], y[idx]


def main() -> None:
    torch.manual_seed(0)
    in_dim = 32
    x, y = make_synthetic_data(in_dim=in_dim)
    x_tr, y_tr = x[:768], y[:768]
    x_va, y_va = x[768:], y[768:]

    model = TinyNet(in_dim=in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    batch_gen = torch.Generator().manual_seed(1)

    epochs = 24
    lr_spike_epoch = 16  # exogenous shock that drives the model into collapse

    # ---- attach the monitor (layers=None -> suggest_layers) ----------------
    # A short baseline over the fast-converging first epochs, then a stable
    # plateau, then the spike: warming_up -> ok -> warning/collapse.
    monitor = CollapseMonitor(model, baseline_epochs=6, z_threshold=3.0, min_signals=2)

    print(f"{'epoch':>5} | {'val_acc':>7} | {'status':>10} | drifting signals")
    print("-" * 66)
    for epoch in range(1, epochs + 1):
        if epoch == lr_spike_epoch:
            for pg in optimizer.param_groups:
                pg["lr"] *= 40.0
            print(f"{'':>5} | {'':>7} | {'':>10} | [lr spike x40 -> inducing collapse]")

        model.train()                       # hooks record
        for x_batch, y_batch in _iterate_batches(x_tr, y_tr, 64, batch_gen):
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()                        # hooks skip (train_only guard)
        with torch.no_grad():
            acc = (model(x_va).argmax(dim=1) == y_va).float().mean().item()

        report = monitor.on_epoch_end(val_acc=acc)
        flag = "  <-- ALERT" if report.alert else ""
        print(f"{report.epoch:>5} | {acc:7.3f} | {report.status:>10} | "
              f"{report.drifting_signals}{flag}")

    monitor.close()
    print("\nDone. Tip: `with CollapseMonitor(model) as monitor:` auto-detaches hooks.")


if __name__ == "__main__":
    main()
