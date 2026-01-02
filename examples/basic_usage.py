"""Basic usage example for mlx-mhc."""

import mlx.core as mx
import mlx.nn as nn
import mlx_mhc as mhc


def demo_sinkhorn():
    """Demonstrate Sinkhorn-Knopp projection."""
    print("=" * 50)
    print("Sinkhorn-Knopp Demo")
    print("=" * 50)

    matrix = mx.random.normal((4, 4))
    print(f"\nInput matrix:\n{matrix}")

    ds = mhc.sinkhorn_knopp(matrix)
    print(f"\nDoubly stochastic result:\n{ds}")
    print(f"\nRow sums (should be ~1): {mx.sum(ds, axis=1)}")
    print(f"Column sums (should be ~1): {mx.sum(ds, axis=0)}")


def demo_mhc_module():
    """Demonstrate ManifoldHyperConnection module."""
    print("\n" + "=" * 50)
    print("ManifoldHyperConnection Demo")
    print("=" * 50)

    batch_size, seq_len, dims = 2, 8, 64
    connection = mhc.ManifoldHyperConnection(dims=dims, expansion=2)

    x = mx.random.normal((batch_size, seq_len, dims))
    layer_output = mx.random.normal((batch_size, seq_len, dims))

    output = connection(x, layer_output)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    h_res = connection._project_h_res()
    print(f"\nH_res (doubly stochastic):\n{h_res}")
    print(f"Row sums: {mx.sum(h_res, axis=1)}")


class SimpleTransformerBlock(nn.Module):
    """Example transformer block using mHC."""

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
        self.mhc_attn = mhc.ManifoldHyperConnection(dims, expansion=2)
        self.mhc_mlp = mhc.ManifoldHyperConnection(dims, expansion=2)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        h = self.mhc_attn(x, attn_out)
        mlp_out = self.mlp(self.norm2(h))
        return self.mhc_mlp(h, mlp_out)


def demo_transformer():
    """Demonstrate mHC in transformer."""
    print("\n" + "=" * 50)
    print("Transformer Integration Demo")
    print("=" * 50)

    block = SimpleTransformerBlock(dims=128)
    x = mx.random.normal((2, 16, 128))
    output = block(x)
    print(f"\nInput: {x.shape} -> Output: {output.shape}")


if __name__ == "__main__":
    demo_sinkhorn()
    demo_mhc_module()
    demo_transformer()
    print("\n" + "=" * 50)
    print("All demos complete!")
