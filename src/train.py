"""
src/train.py
============
Training and evaluation loop utilities.

Functions
---------
train_epoch  : Run one full training epoch, return average loss.
eval_epoch   : Run one full validation epoch (no_grad), return loss + accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model     : The network being trained.
    loader    : DataLoader for the training set.
    optimizer : Optimiser (e.g. SGD, Adam).
    criterion : Loss function (e.g. nn.CrossEntropyLoss()).
                Passed in so it is constructed once per run, not per epoch.
    device    : ``torch.device("cuda")`` or ``torch.device("cpu")``.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in tqdm(loader, desc="  train", leave=False, disable=not show_progress):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[float, float]:
    """
    Evaluate the model for one epoch without gradient tracking.

    Parameters
    ----------
    model     : The network to evaluate.
    loader    : DataLoader for the validation / test set.
    criterion : Loss function - same instance as used in training.
    device    : ``torch.device("cuda")`` or ``torch.device("cpu")``.

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, accuracy)`` where accuracy is in [0, 1].
    """
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for inputs, labels in tqdm(loader, desc="  eval ", leave=False, disable=not show_progress):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss    += loss.item() * batch_size
        preds          = outputs.argmax(dim=1)
        correct       += preds.eq(labels).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / max(1, total_samples)
    accuracy  = correct / total_samples
    return mean_loss, accuracy
