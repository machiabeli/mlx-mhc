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
