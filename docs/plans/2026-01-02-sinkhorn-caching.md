# Sinkhorn Caching for MLX Performance

**Date:** 2026-01-02
**Status:** Approved
**Version target:** 0.4.0

## Problem

Sinkhorn-Knopp runs every forward pass (~20 iterations of matrix ops) to project H_res onto the Birkhoff polytope. During inference, weights are frozen — this computation is wasted.

## Solution

Use MLX's built-in `self.training` property to cache H matrices in eval mode.

### API (No Changes Required)

```python
model.eval()   # All mHC layers cache H matrices
model.train()  # All mHC layers recompute (safe for training)
```

MLX's `nn.Module` automatically propagates `train()`/`eval()` to children.

### Implementation

**1. Add cache attributes in `__init__`:**

```python
self._cached_h_res = None
self._cached_h_pre = None
self._cached_h_post = None
```

**2. New getter methods with caching:**

```python
def _get_h_res(self) -> mx.array:
    if not self.training and self._cached_h_res is not None:
        return self._cached_h_res

    h_res = sinkhorn_knopp(self.h_res_raw, max_iterations=self.sinkhorn_iterations)

    if not self.training:
        self._cached_h_res = h_res
    return h_res
```

**3. Override `train()` to invalidate cache:**

```python
def train(self, mode: bool = True):
    if mode and not self.training:
        self._cached_h_res = None
        self._cached_h_pre = None
        self._cached_h_post = None
    return super().train(mode)
```

**4. Update `pre_scale` and `post_combine` to use `_get_h_*` methods.**

### Paper Alignment

Per [arXiv:2512.24880](https://arxiv.org/abs/2512.24880):

- Default `sinkhorn_iterations` changed from 10 → **20** (paper uses tmax=20)
- Future: kernel fusion targeting 6.7% training overhead

### Tests

1. `test_eval_mode_caches_h_matrices` — verify cache populated after first forward
2. `test_train_mode_clears_cache` — verify switching to train clears cache
3. `test_cache_not_used_in_train_mode` — verify recomputation during training
4. `test_cached_values_match_fresh` — verify cached result equals fresh computation

### Files Changed

- `mlx_mhc/mhc.py` — add caching logic
- `tests/test_mhc.py` — add caching tests
- `mlx_mhc/version.py` — bump to 0.4.0
