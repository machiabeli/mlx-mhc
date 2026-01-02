#!/usr/bin/env python3
"""Benchmark comparing mHC vs standard residual connections."""

import argparse
import time
from mlx_mhc.benchmark import compare_models


def run_benchmark(dims: int, num_layers: int, num_steps: int, batch_size: int, seq_len: int):
    """Run gradient stability benchmark."""
    print("=" * 60)
    print("mHC vs Standard Residuals - Gradient Stability Benchmark")
    print("=" * 60)
    print(f"\nConfig: dims={dims}, layers={num_layers}, steps={num_steps}")
    print(f"        batch={batch_size}, seq_len={seq_len}")
    print("-" * 60)

    start = time.perf_counter()
    results = compare_models(
        dims=dims, num_layers=num_layers, num_steps=num_steps,
        batch_size=batch_size, seq_len=seq_len,
    )
    elapsed = time.perf_counter() - start

    print(f"\n{'Metric':<25} {'Baseline':>15} {'mHC':>15}")
    print("-" * 60)
    print(f"{'Gradient Mean':<25} {results['baseline']['grad_mean']:>15.4f} {results['mhc']['grad_mean']:>15.4f}")
    print(f"{'Gradient Std':<25} {results['baseline']['grad_std']:>15.4f} {results['mhc']['grad_std']:>15.4f}")
    print(f"{'Parameters':<25} {results['baseline']['params']:>15,} {results['mhc']['params']:>15,}")
    print("-" * 60)
    print(f"{'Parameter Overhead':<25} {results['param_overhead_pct']:>15.2f}%")
    print(f"{'Benchmark Time':<25} {elapsed:>15.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC vs standard residuals")
    parser.add_argument("--dims", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    args = parser.parse_args()

    run_benchmark(args.dims, args.layers, args.steps, args.batch, args.seq)


if __name__ == "__main__":
    main()
