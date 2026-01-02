#!/usr/bin/env python3
"""Run mHC vs baseline gradient stability benchmark."""

import argparse
import time
from mlx_mhc.benchmark import compare_models


def main():
    parser = argparse.ArgumentParser(description="mHC vs Baseline Benchmark")
    parser.add_argument("--dims", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    args = parser.parse_args()

    print("=" * 60)
    print("mHC vs Baseline Gradient Stability Benchmark")
    print("=" * 60)
    print(f"Config: dims={args.dims}, layers={args.layers}, steps={args.steps}")

    start = time.perf_counter()
    results = compare_models(
        dims=args.dims, num_layers=args.layers, num_steps=args.steps,
        batch_size=args.batch, seq_len=args.seq
    )
    elapsed = time.perf_counter() - start

    print(f"\nCompleted in {elapsed:.2f}s\n")
    print(f"{'Metric':<20} {'Baseline':>12} {'mHC':>12}")
    print("-" * 45)
    print(f"{'Params':.<20} {results['baseline']['params']:>12,} {results['mhc']['params']:>12,}")
    print(f"{'Grad Mean':.<20} {results['baseline']['grad_mean']:>12.4f} {results['mhc']['grad_mean']:>12.4f}")
    print(f"{'Grad Std':.<20} {results['baseline']['grad_std']:>12.4f} {results['mhc']['grad_std']:>12.4f}")
    print(f"\nParam Overhead: {results['param_overhead_pct']:.2f}%")

    std_ratio = results['mhc']['grad_std'] / max(results['baseline']['grad_std'], 1e-8)
    if std_ratio < 1.0:
        print(f"→ mHC shows {(1-std_ratio)*100:.1f}% more stable gradients")
    else:
        print(f"→ Baseline shows {(std_ratio-1)*100:.1f}% more stable gradients")


if __name__ == "__main__":
    main()
