"""Sinkhorn-Knopp algorithm for projecting matrices to doubly stochastic.

This module provides both fast (GPU-optimized) and legacy (convergence-checking)
implementations of the Sinkhorn-Knopp algorithm for projecting matrices onto
the Birkhoff polytope (doubly stochastic matrices).

Performance Note:
    The fast implementation (default) uses fixed iterations without convergence
    checks, avoiding GPU→CPU synchronization that kills performance. Per the
    mHC paper (arXiv:2512.24880), 20 iterations is sufficient for convergence.
"""

import mlx.core as mx


def sinkhorn_knopp(
    matrix: mx.array,
    max_iterations: int = 20,
    epsilon: float = 1e-6,
    log_space: bool = True,
    check_convergence: bool = False,
) -> mx.array:
    """
    Project a matrix onto the Birkhoff polytope (doubly stochastic matrices).

    Uses the Sinkhorn-Knopp algorithm to iteratively normalize rows and columns
    until both sum to 1.

    Args:
        matrix: Input matrix of shape (n, n). Will be exponentiated internally.
        max_iterations: Number of alternating normalization steps (default: 20).
        epsilon: Convergence threshold (only used if check_convergence=True).
        log_space: If True, use log-space computation for numerical stability.
        check_convergence: If True, check for early convergence (slower due to GPU sync).
            Default False for maximum performance.

    Returns:
        Doubly stochastic matrix where all rows and columns sum to 1.

    Example:
        >>> matrix = mx.random.normal((4, 4))
        >>> ds = sinkhorn_knopp(matrix)
        >>> mx.sum(ds, axis=1)  # All close to 1
        >>> mx.sum(ds, axis=0)  # All close to 1

    Performance:
        With check_convergence=False (default), this function is optimized for
        GPU execution with no CPU synchronization points. The arXiv:2512.24880
        paper shows 20 iterations is sufficient for convergence.
    """
    if check_convergence:
        # Legacy path with convergence checking (slower due to GPU sync)
        if log_space:
            return _sinkhorn_log_space_legacy(matrix, max_iterations, epsilon)
        else:
            return _sinkhorn_direct_legacy(matrix, max_iterations, epsilon)
    else:
        # Fast path - fixed iterations, no sync (default)
        if log_space:
            return _sinkhorn_log_space_fast(matrix, max_iterations)
        else:
            return _sinkhorn_direct_fast(matrix, max_iterations)


# =============================================================================
# FAST IMPLEMENTATIONS (No GPU→CPU sync, fixed iterations)
# =============================================================================


def _sinkhorn_log_space_fast(
    matrix: mx.array,
    iterations: int,
) -> mx.array:
    """
    Fast log-space Sinkhorn with fixed iterations (no convergence check).

    This is the recommended implementation for training - no GPU synchronization
    points means the GPU can execute all iterations without waiting for CPU.
    """
    log_P = matrix

    for _ in range(iterations):
        # Log-space row normalization
        log_P = log_P - mx.logsumexp(log_P, axis=1, keepdims=True)
        # Log-space column normalization
        log_P = log_P - mx.logsumexp(log_P, axis=0, keepdims=True)

    return mx.exp(log_P)


def _sinkhorn_direct_fast(
    matrix: mx.array,
    iterations: int,
) -> mx.array:
    """
    Fast direct Sinkhorn with fixed iterations (no convergence check).

    Less numerically stable than log-space but slightly faster for small matrices.
    """
    P = mx.exp(matrix)
    P = mx.maximum(P, 1e-10)

    for _ in range(iterations):
        P = P / mx.sum(P, axis=1, keepdims=True)
        P = P / mx.sum(P, axis=0, keepdims=True)

    return P


# Compiled versions for additional JIT optimization
@mx.compile
def sinkhorn_knopp_compiled(
    matrix: mx.array,
    iterations: int = 20,
) -> mx.array:
    """
    JIT-compiled Sinkhorn-Knopp for maximum performance.

    MLX's @mx.compile decorator fuses operations and optimizes the computation
    graph. Use this for repeated calls with same-shaped inputs.

    Args:
        matrix: Input matrix of shape (n, n)
        iterations: Number of normalization iterations (default: 20)

    Returns:
        Doubly stochastic matrix
    """
    log_P = matrix

    for _ in range(iterations):
        log_P = log_P - mx.logsumexp(log_P, axis=1, keepdims=True)
        log_P = log_P - mx.logsumexp(log_P, axis=0, keepdims=True)

    return mx.exp(log_P)


# =============================================================================
# LEGACY IMPLEMENTATIONS (With convergence checking - slower but precise)
# =============================================================================


def _sinkhorn_log_space_legacy(
    matrix: mx.array,
    max_iterations: int,
    epsilon: float,
) -> mx.array:
    """
    Log-space Sinkhorn with convergence checking (legacy).

    WARNING: The float() calls cause GPU→CPU synchronization each iteration,
    significantly reducing performance. Use _sinkhorn_log_space_fast instead.
    """
    log_P = matrix

    for _ in range(max_iterations):
        log_P = log_P - mx.logsumexp(log_P, axis=1, keepdims=True)
        log_P = log_P - mx.logsumexp(log_P, axis=0, keepdims=True)

        # Convergence check (causes GPU sync - slow!)
        P = mx.exp(log_P)
        row_sums = mx.sum(P, axis=1)
        col_sums = mx.sum(P, axis=0)

        row_err = mx.max(mx.abs(row_sums - 1.0))
        col_err = mx.max(mx.abs(col_sums - 1.0))

        if float(row_err) < epsilon and float(col_err) < epsilon:
            break

    return mx.exp(log_P)


def _sinkhorn_direct_legacy(
    matrix: mx.array,
    max_iterations: int,
    epsilon: float,
) -> mx.array:
    """
    Direct Sinkhorn with convergence checking (legacy).

    WARNING: The float() calls cause GPU→CPU synchronization each iteration,
    significantly reducing performance. Use _sinkhorn_direct_fast instead.
    """
    P = mx.exp(matrix)
    P = mx.maximum(P, 1e-10)

    for _ in range(max_iterations):
        P = P / mx.sum(P, axis=1, keepdims=True)
        P = P / mx.sum(P, axis=0, keepdims=True)

        # Convergence check (causes GPU sync - slow!)
        row_sums = mx.sum(P, axis=1)
        col_sums = mx.sum(P, axis=0)

        row_err = mx.max(mx.abs(row_sums - 1.0))
        col_err = mx.max(mx.abs(col_sums - 1.0))

        if float(row_err) < epsilon and float(col_err) < epsilon:
            break

    return P
