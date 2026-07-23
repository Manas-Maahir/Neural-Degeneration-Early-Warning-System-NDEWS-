"""
ndews/signals.py
================
PyTorch hook infrastructure for extracting internal model signals during
training. Metrics are recorded per forward/backward pass and averaged
at epoch end by ``SignalLogger``.

Metrics
-------
- Representation effective rank  (replaces softmax entropy — theoretically grounded)
- Gradient diversity
- Feature reuse
- Neuron sparsity
- Representational isotropy      (replaces redundant embedding variance)
- Activation scale

Usage
-----
    logger = SignalLogger(model, target_layers=["conv2", "fc1"])
    for epoch in range(n_epochs):
        logger.reset()
        train_epoch(model, ...)
        signals = logger.get_epoch_signals()
    logger.remove_hooks()

Signal keys returned by get_epoch_signals()
-------------------------------------------
Each key has the form ``{layer_name}{suffix}`` where suffix is one of:

    _representation_entropy   effective rank of the activation matrix
    _feature_reuse            mean off-diagonal cosine similarity (Gram)
    _gradient_diversity       variance of output gradients across the batch
    _neuron_sparsity          fraction of near-zero activations (adaptive threshold)
    _representational_isotropy  uniformity of per-dimension variance
    _activation_scale         global RMS scale of activations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-level canonical suffix constants
# ---------------------------------------------------------------------------

CANONICAL_METRIC_SUFFIXES: tuple[str, ...] = (
    "_representation_entropy",
    "_feature_reuse",
    "_gradient_diversity",
    "_neuron_sparsity",
    "_representational_isotropy",
    "_activation_scale",
)


# ---------------------------------------------------------------------------
# Shared tensor helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hook implementations
# ---------------------------------------------------------------------------

class RepresentationEntropyHook:
    """
    Forward hook measuring the effective rank of the activation matrix.

    Effective rank (Roy & Vetterli, 2007) is defined as:

        R_eff(A) = exp( H( σ / Σσ ) )

    where σ are the singular values of the mean-centred activation matrix
    A ∈ R^{B × D}.  The result lies in [1, min(B, D)]:

    - Close to 1  → representations collapsed onto ~1 dimension.
    - Close to D  → representations maximally spread across all dimensions.

    This replaces the previous softmax-entropy formulation, which was
    sensitive to activation *scale* rather than representational *diversity*.
    """

    def __init__(self) -> None:
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        emb = _pool_to_embeddings(output.detach())  # [B, D]
        B, D = emb.shape[0], emb.shape[1] if emb.dim() > 1 else 1
        if B < 2 or D < 2:
            return

        emb = emb - emb.mean(dim=0, keepdim=True)  # centre columns

        try:
            S = torch.linalg.svdvals(emb)  # [min(B, D)], descending
        except RuntimeError:
            return

        S = S.clamp(min=0.0)
        total = S.sum()
        if total < 1e-9:
            return

        p = S / total
        # Shannon entropy in nats, then exponentiate → effective rank
        H = -(p * torch.log(p + 1e-9)).sum()
        self._values.append(torch.exp(H).item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class FeatureReuseDetector:
    """
    Forward hook for average off-diagonal cosine similarity of feature columns.

    Computes a Gram-matrix of centred, L2-normalised feature columns.  The
    mean absolute off-diagonal entry measures how redundant (linearly
    dependent) learned features are — high values indicate collapse to a
    low-diversity feature set.
    """

    def __init__(self) -> None:
        self._scores: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        out = _pool_to_embeddings(output.detach())  # [B, D]
        if out.size(0) < 2:
            return

        centered = out - out.mean(dim=0, keepdim=True)
        norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
        normalized = centered / norms  # [B, D], unit-norm columns

        # Gram matrix of cosine similarities between feature columns.
        # Shape [D, D]; diagonal = 1.0 (self-similarity), zeroed out below.
        corr = torch.mm(normalized.T, normalized)
        corr.fill_diagonal_(0.0)
        self._scores.append(corr.abs().mean().item())

    def reset(self) -> None:
        self._scores.clear()

    def get_metric(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0


class GradientDiversityTracker:
    """
    Backward hook measuring variance of output gradients across the batch.

    Low variance indicates all samples produce near-identical gradient
    signals — a sign that the network's training signal has become
    monolithic and uninformative.
    """

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

        self._variances.append(g.var(dim=0, unbiased=False).mean().item())

    def reset(self) -> None:
        self._variances.clear()

    def get_metric(self) -> float:
        return sum(self._variances) / len(self._variances) if self._variances else 0.0


class NeuronSparsityTracker:
    """
    Forward hook measuring fraction of near-zero activations.

    Uses an *adaptive* threshold: ``scale * relative_threshold``, where
    ``scale`` is the per-batch mean absolute activation value.  This makes
    the metric invariant to the layer's activation range — a fixed absolute
    threshold (e.g. 1e-3) would report misleadingly high sparsity when
    activations are naturally small-scale, or misleadingly low sparsity for
    large-scale activations.

    Parameters
    ----------
    relative_threshold : float
        Fraction of the mean absolute activation below which a unit is
        considered inactive.  Default 0.01 (1 % of mean scale).
    threshold_mode : str
        ``"adaptive"`` (default) uses relative_threshold × mean|act|.
        ``"absolute"`` uses relative_threshold directly as a fixed cutoff.
    """

    def __init__(
        self,
        relative_threshold: float = 0.01,
        threshold_mode: str = "adaptive",
    ) -> None:
        if threshold_mode not in ("adaptive", "absolute"):
            raise ValueError(f"threshold_mode must be 'adaptive' or 'absolute', got {threshold_mode!r}")
        self.relative_threshold = relative_threshold
        self.threshold_mode = threshold_mode
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

        if self.threshold_mode == "adaptive":
            scale = out.abs().mean()
            threshold = (scale * self.relative_threshold).clamp(min=1e-9)
        else:
            threshold = self.relative_threshold

        sparsity = (out.abs() < threshold).float().mean()
        self._values.append(sparsity.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class RepresentationalIsotropyTracker:
    """
    Forward hook measuring uniformity of per-dimension activation variance.

    Isotropy is defined as:

        isotropy = mean_dim_variance / max_dim_variance

    A value of 1.0 means all dimensions contribute equally (isotropic,
    maximally spread).  A value approaching 0 means variance is concentrated
    in a small number of dimensions — a form of dimensional collapse distinct
    from the scale changes captured by ActivationScaleTracker.

    Replaces the previous EmbeddingVarianceTracker, which measured the same
    underlying quantity as ActivationScaleTracker with a different aggregation.
    """

    def __init__(self) -> None:
        self._values: list[float] = []

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        emb = _pool_to_embeddings(output.detach())  # [B, D]
        if emb.numel() == 0 or emb.size(0) < 2 or emb.size(1) < 2:
            return

        per_dim_var = emb.var(dim=0, unbiased=False)  # [D]
        max_var = per_dim_var.max()
        if max_var < 1e-9:
            return
        isotropy = per_dim_var.mean() / max_var
        self._values.append(isotropy.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class ActivationScaleTracker:
    """
    Forward hook for global RMS activation scale.

    Tracks whether activation magnitudes are growing or vanishing — a
    precursor to gradient explosion or saturation.  Renamed from
    ActivationVarianceTracker to clarify its role as a scale monitor
    (RMS is more interpretable than raw variance for this purpose).
    """

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
        rms = out.pow(2).mean().sqrt()
        self._values.append(rms.item())

    def reset(self) -> None:
        self._values.clear()

    def get_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


# ---------------------------------------------------------------------------
# Signal logger
# ---------------------------------------------------------------------------

class SignalLogger:
    """
    Register and manage all signal hooks on specified model layers.

    Parameters
    ----------
    model : nn.Module
        Model to instrument.
    target_layers : list[str]
        Layer names from ``model.named_modules()``.
    """

    def __init__(self, model: nn.Module, target_layers: list[str]) -> None:
        self.model = model
        self.target_layers = target_layers

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entropy: dict[str, RepresentationEntropyHook] = {}
        self._reuse: dict[str, FeatureReuseDetector] = {}
        self._grad: dict[str, GradientDiversityTracker] = {}
        self._sparsity: dict[str, NeuronSparsityTracker] = {}
        self._isotropy: dict[str, RepresentationalIsotropyTracker] = {}
        self._scale: dict[str, ActivationScaleTracker] = {}

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
            i_hook = RepresentationalIsotropyTracker()
            a_hook = ActivationScaleTracker()

            self._handles += [
                module.register_forward_hook(e_hook),
                module.register_forward_hook(r_hook),
                module.register_full_backward_hook(g_hook),
                module.register_forward_hook(s_hook),
                module.register_forward_hook(i_hook),
                module.register_forward_hook(a_hook),
            ]

            self._entropy[name] = e_hook
            self._reuse[name] = r_hook
            self._grad[name] = g_hook
            self._sparsity[name] = s_hook
            self._isotropy[name] = i_hook
            self._scale[name] = a_hook
            self._registered.add(name)

    def reset(self) -> None:
        """Clear accumulated values at epoch boundary."""
        for name in self._registered:
            self._entropy[name].reset()
            self._reuse[name].reset()
            self._grad[name].reset()
            self._sparsity[name].reset()
            self._isotropy[name].reset()
            self._scale[name].reset()

    def get_epoch_signals(self) -> dict[str, float]:
        """
        Return averaged epoch signals as a flat dict.

        Keys (one set per tracked layer):
        - ``{layer}_representation_entropy``   effective rank ∈ [1, min(B,D)]
        - ``{layer}_feature_reuse``            mean off-diagonal Gram similarity
        - ``{layer}_gradient_diversity``       mean per-dim gradient variance
        - ``{layer}_neuron_sparsity``          fraction of near-zero activations
        - ``{layer}_representational_isotropy``  mean/max per-dim variance ratio
        - ``{layer}_activation_scale``         global RMS activation magnitude
        """
        signals: dict[str, float] = {}
        for name in self._registered:
            signals[f"{name}_representation_entropy"] = self._entropy[name].get_metric()
            signals[f"{name}_feature_reuse"]          = self._reuse[name].get_metric()
            signals[f"{name}_gradient_diversity"]     = self._grad[name].get_metric()
            signals[f"{name}_neuron_sparsity"]        = self._sparsity[name].get_metric()
            signals[f"{name}_representational_isotropy"] = self._isotropy[name].get_metric()
            signals[f"{name}_activation_scale"]       = self._scale[name].get_metric()
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
                if any(k.endswith(suffix) for suffix in CANONICAL_METRIC_SUFFIXES)
            }

        if not signals:
            return "[SignalLogger] No signals recorded yet."

        lines = ["[SignalLogger] Epoch signals:"]
        for key, value in sorted(signals.items()):
            lines.append(f"  {key:<48s} = {value:.6f}")
        return "\n".join(lines)
