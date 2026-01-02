"""Manifold-Constrained Hyper-Connections module."""

import math
import mlx.core as mx
import mlx.nn as nn

from .sinkhorn import sinkhorn_knopp


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) module.

    Implements the mHC architecture from DeepSeek's paper (arXiv:2512.24880).
    Projects residual connections onto a manifold to maintain training stability.

    The forward pass computes:
        output = H_post * (layer_output + H_res * H_pre * x)

    Where:
        - H_res is constrained to be doubly stochastic (Birkhoff polytope)
        - H_pre and H_post are constrained to be non-negative via sigmoid

    Args:
        dims: Hidden dimension of the input/output.
        expansion: Expansion factor for the hyper-connection width (default: 2).
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations (default: 10).
    """

    def __init__(
        self,
        dims: int,
        expansion: int = 2,
        sinkhorn_iterations: int = 10,
    ):
        super().__init__()

        self.dims = dims
        self.expansion = expansion
        self.sinkhorn_iterations = sinkhorn_iterations

        # Learnable parameters (raw, before projection)
        scale = 1.0 / math.sqrt(expansion)
        self.h_res_raw = mx.random.normal((expansion, expansion)) * scale
        self.h_pre_raw = mx.zeros((expansion,))
        self.h_pre_bias = mx.zeros((expansion,))
        self.h_post_raw = mx.zeros((expansion,))
        self.h_post_bias = mx.zeros((expansion,))

    def _project_h_res(self) -> mx.array:
        """Project H_res to doubly stochastic matrix using Sinkhorn-Knopp."""
        return sinkhorn_knopp(
            self.h_res_raw,
            max_iterations=self.sinkhorn_iterations,
            log_space=True,
        )

    def _project_h_pre(self) -> mx.array:
        """Project H_pre to non-negative via sigmoid."""
        return mx.sigmoid(self.h_pre_raw + self.h_pre_bias)

    def _project_h_post(self) -> mx.array:
        """Project H_post to non-negative [0, 2] via scaled sigmoid."""
        return 2.0 * mx.sigmoid(self.h_post_raw + self.h_post_bias)

    def __call__(self, x: mx.array, layer_output: mx.array) -> mx.array:
        """
        Apply manifold-constrained hyper-connection.

        Args:
            x: Input tensor of shape (batch, seq_len, dims)
            layer_output: Output from the layer of shape (batch, seq_len, dims)

        Returns:
            Output tensor of shape (batch, seq_len, dims)
        """
        batch_size, seq_len, dims = x.shape

        h_res = self._project_h_res()
        h_pre = self._project_h_pre()
        h_post = self._project_h_post()

        # Reshape for expansion: (batch, seq, dims) -> (batch, seq, expansion, dims//expansion)
        x_expanded = x.reshape(batch_size, seq_len, self.expansion, -1)

        # Apply H_pre
        x_pre = x_expanded * h_pre.reshape(1, 1, self.expansion, 1)

        # Apply H_res via einsum
        x_res = mx.einsum('ij,...jd->...id', h_res, x_pre)

        # Reshape layer_output
        layer_expanded = layer_output.reshape(batch_size, seq_len, self.expansion, -1)

        # Combine and apply H_post
        combined = layer_expanded + x_res
        output_expanded = combined * h_post.reshape(1, 1, self.expansion, 1)

        return output_expanded.reshape(batch_size, seq_len, dims)
