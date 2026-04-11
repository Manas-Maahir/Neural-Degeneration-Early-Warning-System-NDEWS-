"""
src/signals.py
==============
PyTorch hook infrastructure for extracting internal model signals during
training. Metrics are recorded per forward/backward pass and averaged
at epoch end by ``SignalLogger``.

Metrics
-------
- Representation entropy
- Gradient diversity
- Feature reuse
- Neuron sparsity
- Embedding variance
- Activation variance

Usage
-----
    logger = SignalLogger(model, target_layers=["conv2", "fc1"])
    for epoch in range(n_epochs):
        logger.reset()
        train_epoch(model, ...)
        signals = logger.get_epoch_signals()
    logger.remove_hooks()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions into one feature axis."""
    return x.reshape(x.size(0), -1)


def _pool_to_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    Convert activations to a [B, D] embedding tensor.
    Conv activations are globally averaged over spatial dimensions.
    """
    if x.dim() <= 2:
        return x
    reduce_dims = tuple(range(2, x.dim()))
    return x.mean(dim=reduce_dims)


class RepresentationEntropyHook:
    """Forward hook for mean Shannon entropy of activations."""

    def __init__(self) -> None:
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        out = output.detach()
        if out.size(0) == 0:
            return
        flat = _flatten_batch(out)
        p = F.softmax(flat, dim=-1)
        entropy = -(p * torch.log2(p + 1e-9)).sum(dim=-1).mean()
        self._values.append(entropy.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class FeatureReuseDetector:
    """Forward hook for average inter-feature correlation magnitude."""

    def __init__(self) -> None:
        self._scores: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        out = _pool_to_embeddings(output.detach())
        if out.size(0) < 2:
            return

        centered = out - out.mean(dim=0, keepdim=True)
        norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
        normalized = centered / norms

        corr = torch.mm(normalized.T, normalized) / out.size(0)
        corr.fill_diagonal_(0.0)
        self._scores.append(corr.abs().mean().item())

    def reset(self) -> None:
        self._scores.clear()

    def get_metric(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0


class GradientDiversityTracker:
    """Backward hook for mean variance of gradients across the batch."""

    def __init__(self) -> None:
        self._variances: list[float] = []

    def __call__(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        grads = grad_output[0]
        if grads is None:
            return

        g = grads.detach()
        if g.size(0) == 0:
            return
        if g.dim() > 2:
            g = _flatten_batch(g)

        grad_diversity = g.var(dim=0, unbiased=False).mean()
        self._variances.append(grad_diversity.item())

    def reset(self) -> None:
        self._variances.clear()

    def get_metric(self) -> float:
        return sum(self._variances) / len(self._variances) if self._variances else 0.0


class NeuronSparsityTracker:
    """Forward hook for fraction of near-zero activations."""

    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        out = output.detach()
        if out.numel() == 0:
            return
        sparsity = (out.abs() < self.threshold).float().mean()
        self._values.append(sparsity.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class EmbeddingVarianceTracker:
    """Forward hook for variance of pooled embeddings across the batch."""

    def __init__(self) -> None:
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        emb = _pool_to_embeddings(output.detach())
        if emb.numel() == 0:
            return
        variance = emb.var(dim=0, unbiased=False).mean()
        self._values.append(variance.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class ActivationVarianceTracker:
    """Forward hook for global activation variance."""

    def __init__(self) -> None:
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        out = output.detach()
        if out.numel() == 0:
            return
        variance = out.var(unbiased=False)
        self._values.append(variance.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class SignalLogger:
    """
    Register and manage signal hooks on specified model layers.

    Parameters
    ----------
    model : nn.Module
        Model to instrument.
    target_layers : list[str]
        Layer names from ``model.named_modules()``.
    """

    _CANONICAL_SUFFIXES = (
        "_representation_entropy",
        "_feature_reuse",
        "_gradient_diversity",
        "_neuron_sparsity",
        "_embedding_variance",
        "_activation_variance",
    )

    def __init__(self, model: nn.Module, target_layers: list[str]) -> None:
        self.model = model
        self.target_layers = target_layers

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entropy: dict[str, RepresentationEntropyHook] = {}
        self._reuse: dict[str, FeatureReuseDetector] = {}
        self._grad: dict[str, GradientDiversityTracker] = {}
        self._sparsity: dict[str, NeuronSparsityTracker] = {}
        self._embed_var: dict[str, EmbeddingVarianceTracker] = {}
        self._act_var: dict[str, ActivationVarianceTracker] = {}

        self._registered: set[str] = set()
        self._register_hooks()

        missing = set(target_layers) - self._registered
        if missing:
            print(f"[SignalLogger] WARNING: layers not found in model: {sorted(missing)}")

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name not in self.target_layers:
                continue

            e_hook = RepresentationEntropyHook()
            r_hook = FeatureReuseDetector()
            g_hook = GradientDiversityTracker()
            s_hook = NeuronSparsityTracker()
            v_hook = EmbeddingVarianceTracker()
            a_hook = ActivationVarianceTracker()

            self._handles += [
                module.register_forward_hook(e_hook),
                module.register_forward_hook(r_hook),
                module.register_full_backward_hook(g_hook),
                module.register_forward_hook(s_hook),
                module.register_forward_hook(v_hook),
                module.register_forward_hook(a_hook),
            ]

            self._entropy[name] = e_hook
            self._reuse[name] = r_hook
            self._grad[name] = g_hook
            self._sparsity[name] = s_hook
            self._embed_var[name] = v_hook
            self._act_var[name] = a_hook
            self._registered.add(name)

    def reset(self) -> None:
        """Clear accumulated values at epoch boundary."""
        for name in self._registered:
            self._entropy[name].reset()
            self._reuse[name].reset()
            self._grad[name].reset()
            self._sparsity[name].reset()
            self._embed_var[name].reset()
            self._act_var[name].reset()

    def get_epoch_signals(self) -> dict[str, float]:
        """
        Return averaged epoch signals as a flat dict.

        Canonical keys:
        - ``{layer}_representation_entropy``
        - ``{layer}_feature_reuse``
        - ``{layer}_gradient_diversity``
        - ``{layer}_neuron_sparsity``
        - ``{layer}_embedding_variance``
        - ``{layer}_activation_variance``

        Backward-compatible aliases are also included:
        - ``{layer}_entropy``
        - ``{layer}_reuse``
        - ``{layer}_grad_var``
        - ``{layer}_grad_div``
        - ``{layer}_sparsity``
        - ``{layer}_embed_var``
        - ``{layer}_act_var``
        """
        signals: dict[str, float] = {}
        for name in self._registered:
            entropy = self._entropy[name].get_metric()
            reuse = self._reuse[name].get_metric()
            grad_div = self._grad[name].get_metric()
            sparsity = self._sparsity[name].get_metric()
            embed_var = self._embed_var[name].get_metric()
            act_var = self._act_var[name].get_metric()

            # Canonical metric names
            signals[f"{name}_representation_entropy"] = entropy
            signals[f"{name}_feature_reuse"] = reuse
            signals[f"{name}_gradient_diversity"] = grad_div
            signals[f"{name}_neuron_sparsity"] = sparsity
            signals[f"{name}_embedding_variance"] = embed_var
            signals[f"{name}_activation_variance"] = act_var

            # Backward-compatible aliases
            signals[f"{name}_entropy"] = entropy
            signals[f"{name}_reuse"] = reuse
            signals[f"{name}_grad_var"] = grad_div
            signals[f"{name}_grad_div"] = grad_div
            signals[f"{name}_sparsity"] = sparsity
            signals[f"{name}_embed_var"] = embed_var
            signals[f"{name}_act_var"] = act_var

        return signals

    def remove_hooks(self) -> None:
        """De-register all hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self, canonical_only: bool = True) -> str:
        """Return a human-readable snapshot of the current epoch signals."""
        signals = self.get_epoch_signals()
        if canonical_only:
            signals = {
                k: v
                for k, v in signals.items()
                if any(k.endswith(suffix) for suffix in self._CANONICAL_SUFFIXES)
            }

        if not signals:
            return "[SignalLogger] No signals recorded yet."

        lines = ["[SignalLogger] Epoch signals:"]
        for key, value in sorted(signals.items()):
            lines.append(f"  {key:<40s} = {value:.6f}")
        return "\n".join(lines)
