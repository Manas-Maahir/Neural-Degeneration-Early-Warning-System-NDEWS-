"""Tests for suggest_layers (ndews/monitor.py)."""

from __future__ import annotations

import torch.nn as nn

from ndews import suggest_layers


def test_conv_net_picks_last_conv_and_last_linear(tiny_conv):
    assert suggest_layers(tiny_conv) == ["conv2", "fc"]


def test_mlp_picks_last_linear_only(tiny_mlp):
    assert suggest_layers(tiny_mlp) == ["fc2"]


def test_no_conv_or_linear_returns_empty():
    model = nn.Sequential(nn.ReLU(), nn.Dropout())
    assert suggest_layers(model) == []
