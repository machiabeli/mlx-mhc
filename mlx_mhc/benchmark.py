"""Benchmark utilities for comparing mHC vs standard residuals."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Any

from .mhc import ManifoldHyperConnection


class GradientTracker:
    """Track gradient norms during training."""

    def __init__(self):
        self.history: List[float] = []

    def record(self, grads: Dict[str, mx.array]) -> None:
        """Record the total gradient norm for this step."""
        total_norm_sq = 0.0
        for name, grad in grads.items():
            if isinstance(grad, mx.array):
                norm_sq = float(mx.sum(grad ** 2))
                total_norm_sq += norm_sq
        self.history.append(total_norm_sq ** 0.5)

    def stats(self) -> Dict[str, float]:
        """Compute mean and std of recorded gradient norms."""
        if not self.history:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        return {"mean": mean, "std": variance ** 0.5}


class BaselineBlock(nn.Module):
    """Transformer block with standard residual connections."""

    def __init__(self, dims: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Linear(dims * 4, dims),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        return h + self.mlp(self.norm2(h))


class MHCBlock(nn.Module):
    """Transformer block with mHC residual connections."""

    def __init__(self, dims: int, num_heads: int = 4, expansion: int = 2):
        super().__init__()
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Linear(dims * 4, dims),
        )
        self.mhc_attn = ManifoldHyperConnection(dims, expansion)
        self.mhc_mlp = ManifoldHyperConnection(dims, expansion)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        h = self.mhc_attn(x, attn_out)
        mlp_out = self.mlp(self.norm2(h))
        return self.mhc_mlp(h, mlp_out)


class BaselineModel(nn.Module):
    """Multi-layer transformer with standard residuals."""

    def __init__(self, dims: int, num_layers: int, num_heads: int = 4):
        super().__init__()
        self.layers = [BaselineBlock(dims, num_heads) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MHCModel(nn.Module):
    """Multi-layer transformer with mHC residuals."""

    def __init__(self, dims: int, num_layers: int, num_heads: int = 4, expansion: int = 2):
        super().__init__()
        self.layers = [MHCBlock(dims, num_heads, expansion) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def create_baseline_model(dims: int, num_layers: int, num_heads: int = 4) -> BaselineModel:
    """Create transformer with standard residual connections."""
    return BaselineModel(dims, num_layers, num_heads)


def create_mhc_model(dims: int, num_layers: int, num_heads: int = 4, expansion: int = 2) -> MHCModel:
    """Create transformer with mHC residual connections."""
    return MHCModel(dims, num_layers, num_heads, expansion)


def _count_recursive(obj: Any) -> int:
    """Recursively count parameters in nested structure."""
    if isinstance(obj, mx.array):
        return obj.size
    elif isinstance(obj, dict):
        return sum(_count_recursive(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(_count_recursive(v) for v in obj)
    return 0


def count_params(model: nn.Module) -> int:
    """Count total parameters in model."""
    return _count_recursive(model.parameters())


def train_step(model: nn.Module, x: mx.array, y: mx.array) -> tuple[mx.array, float]:
    """Execute single training step, return loss and gradient norm."""

    def loss_fn(model):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss)

    # Compute gradient norm
    grad_norm = _count_grad_norm(grads)

    return loss, grad_norm


def _count_grad_norm(grads: Any) -> float:
    """Recursively compute gradient norm."""
    if isinstance(grads, mx.array):
        return float(mx.sum(grads ** 2))
    elif isinstance(grads, dict):
        return sum(_count_grad_norm(v) for v in grads.values())
    elif isinstance(grads, (list, tuple)):
        return sum(_count_grad_norm(v) for v in grads)
    return 0.0


def _compute_grad_norm(grads: Any) -> float:
    """Compute total gradient norm from nested gradient structure."""
    if isinstance(grads, mx.array):
        return float(mx.sum(grads ** 2))
    elif isinstance(grads, dict):
        return sum(_compute_grad_norm(v) for v in grads.values())
    elif isinstance(grads, (list, tuple)):
        return sum(_compute_grad_norm(v) for v in grads)
    return 0.0


def train_step(model: nn.Module, x: mx.array, y: mx.array) -> tuple[mx.array, float]:
    """Perform a single training step, return loss and gradient norm."""
    def loss_fn(params):
        model.update(params)
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(loss, grads)
    grad_norm = _compute_grad_norm(grads) ** 0.5
    return loss, grad_norm


def compare_models(
    dims: int,
    num_layers: int,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    num_heads: int = 4,
    expansion: int = 2,
) -> Dict[str, Any]:
    """Compare gradient stability between baseline and mHC models.
    
    Returns dict with 'baseline' and 'mhc' stats plus param overhead.
    """
    # Create models
    baseline = create_baseline_model(dims, num_layers, num_heads)
    mhc = create_mhc_model(dims, num_layers, num_heads, expansion)
    
    # Track gradients
    baseline_tracker = GradientTracker()
    mhc_tracker = GradientTracker()
    
    # Run training steps
    for _ in range(num_steps):
        x = mx.random.normal((batch_size, seq_len, dims))
        y = mx.random.normal((batch_size, seq_len, dims))
        
        # Baseline
        _, grad_norm = train_step(baseline, x, y)
        baseline_tracker.history.append(grad_norm ** 0.5)
        
        # mHC
        _, grad_norm = train_step(mhc, x, y)
        mhc_tracker.history.append(grad_norm ** 0.5)
    
    # Compute stats
    baseline_stats = baseline_tracker.stats()
    mhc_stats = mhc_tracker.stats()
    
    # Param overhead
    baseline_params = count_params(baseline)
    mhc_params = count_params(mhc)
    overhead = ((mhc_params - baseline_params) / baseline_params) * 100
    
    return {
        "baseline": {
            "grad_mean": baseline_stats["mean"],
            "grad_std": baseline_stats["std"],
            "params": baseline_params,
        },
        "mhc": {
            "grad_mean": mhc_stats["mean"],
            "grad_std": mhc_stats["std"],
            "params": mhc_params,
        },
        "param_overhead_pct": overhead,
    }


def compare_models(
    dims: int,
    num_layers: int,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    num_heads: int = 4,
    expansion: int = 2,
) -> Dict[str, Any]:
    """
    Compare gradient stability between baseline and mHC models.

    Runs training for num_steps and tracks gradient norms for both models.

    Returns:
        Dict with 'baseline', 'mhc' stats and 'param_overhead_pct'
    """
    # Create models
    baseline = create_baseline_model(dims, num_layers, num_heads)
    mhc = create_mhc_model(dims, num_layers, num_heads, expansion)

    # Track gradients
    baseline_tracker = GradientTracker()
    mhc_tracker = GradientTracker()

    # Run training steps
    for _ in range(num_steps):
        x = mx.random.normal((batch_size, seq_len, dims))
        y = mx.random.normal((batch_size, seq_len, dims))

        # Baseline step
        _, grad_norm = train_step(baseline, x, y)
        baseline_tracker.history.append(grad_norm)

        # mHC step (same data for fair comparison)
        _, grad_norm = train_step(mhc, x, y)
        mhc_tracker.history.append(grad_norm)

    # Compute stats
    baseline_stats = baseline_tracker.stats()
    mhc_stats = mhc_tracker.stats()

    # Compute param overhead
    baseline_params = count_params(baseline)
    mhc_params = count_params(mhc)
    overhead_pct = ((mhc_params - baseline_params) / baseline_params) * 100

    return {
        "baseline": {
            "grad_mean": baseline_stats["mean"],
            "grad_std": baseline_stats["std"],
            "params": baseline_params,
        },
        "mhc": {
            "grad_mean": mhc_stats["mean"],
            "grad_std": mhc_stats["std"],
            "params": mhc_params,
        },
        "param_overhead_pct": overhead_pct,
    }
