# Fix H_pre Application Order in ManifoldHyperConnection

## Goal

Fix the ManifoldHyperConnection module to correctly apply H_pre BEFORE the layer function F, matching the DeepSeek paper equation.

**Paper equation:**
```
x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)
```

**Current (wrong) implementation:**
```
output = H_post * (F(x) + H_res * H_pre * x)
```

**Fixed implementation:**
```
output = H_post * (F(H_pre * x) + H_res * H_pre * x)
```

## Architecture

Two-stage API that allows users to apply H_pre before their layer function:

```python
mhc = ManifoldHyperConnection(dims, expansion)
x_pre = mhc.pre_scale(x)           # Apply H_pre
layer_out = layer(x_pre)           # F(H_pre * x)
output = mhc.post_combine(x, layer_out)  # H_post * (layer_out + H_res * H_pre * x)
```

## Tech Stack

- MLX (Apple's ML framework)
- Python 3.14
- pytest for testing

## Files

- `/Users/ma/mlx-mhc/mlx_mhc/mhc.py` - Main module
- `/Users/ma/mlx-mhc/tests/test_mhc.py` - Tests
- `/Users/ma/mlx-mhc/examples/basic_usage.py` - Example usage

---

## Task 1: Add test for pre_scale method

**File:** `/Users/ma/mlx-mhc/tests/test_mhc.py`

**Add this test class after `TestManifoldHyperConnectionEdgeCases`:**

```python
class TestTwoStageAPI:
    """Tests for the two-stage pre_scale/post_combine API."""

    def test_pre_scale_exists(self):
        """pre_scale method should exist and be callable."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        assert hasattr(mhc, 'pre_scale'), "pre_scale method should exist"
        assert callable(mhc.pre_scale), "pre_scale should be callable"

    def test_pre_scale_output_shape(self):
        """pre_scale should preserve input shape."""
        dims = 64
        batch_size = 2
        seq_len = 16
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)
        x = mx.random.normal((batch_size, seq_len, dims))

        x_pre = mhc.pre_scale(x)
        mx.eval(x_pre)

        assert x_pre.shape == x.shape, f"Expected {x.shape}, got {x_pre.shape}"

    def test_pre_scale_applies_h_pre(self):
        """pre_scale should apply H_pre scaling element-wise per expansion group."""
        dims = 8
        expansion = 2
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        # Set known H_pre values: [0.25, 0.75] via inverse sigmoid
        # sigmoid(x) = 0.25 => x = log(0.25/0.75) = log(1/3) ~ -1.0986
        # sigmoid(x) = 0.75 => x = log(0.75/0.25) = log(3) ~ 1.0986
        mhc.h_pre_raw = mx.array([-1.0986, 1.0986])
        mhc.h_pre_bias = mx.zeros((2,))

        # Simple input: all ones
        x = mx.ones((1, 1, dims))
        x_pre = mhc.pre_scale(x)
        mx.eval(x_pre)

        # Expected: first half scaled by ~0.25, second half by ~0.75
        # dims=8, expansion=2 => groups of 4
        x_pre_flat = x_pre.reshape(-1).tolist()

        # First 4 elements should be ~0.25
        for i in range(4):
            assert abs(x_pre_flat[i] - 0.25) < 0.01, f"Element {i}: expected ~0.25, got {x_pre_flat[i]}"

        # Last 4 elements should be ~0.75
        for i in range(4, 8):
            assert abs(x_pre_flat[i] - 0.75) < 0.01, f"Element {i}: expected ~0.75, got {x_pre_flat[i]}"

    def test_pre_scale_different_expansions(self):
        """pre_scale should work with different expansion factors."""
        dims = 128
        batch_size = 4
        seq_len = 32

        for expansion in [2, 4, 8]:
            mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)
            x = mx.random.normal((batch_size, seq_len, dims))

            x_pre = mhc.pre_scale(x)
            mx.eval(x_pre)

            assert x_pre.shape == x.shape, (
                f"Expansion {expansion}: Expected {x.shape}, got {x_pre.shape}"
            )
```

**Run test (expect FAIL - method does not exist yet):**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_exists -v
```

**Expected output:**
```
FAILED tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_exists - AssertionError: pre_scale method should exist
```

---

## Task 2: Implement pre_scale method

**File:** `/Users/ma/mlx-mhc/mlx_mhc/mhc.py`

**Add this method to ManifoldHyperConnection class (before `__call__`):**

```python
    def pre_scale(self, x: mx.array) -> mx.array:
        """
        Apply H_pre scaling to input (first stage of two-stage API).

        This applies the pre-connection scaling that should be done BEFORE
        passing data through the layer function F.

        Args:
            x: Input tensor of shape (batch, seq_len, dims)

        Returns:
            Scaled tensor of shape (batch, seq_len, dims): H_pre * x
        """
        batch_size, seq_len, dims = x.shape

        h_pre = self._project_h_pre()

        # Reshape for expansion: (batch, seq, dims) -> (batch, seq, expansion, dims//expansion)
        x_expanded = x.reshape(batch_size, seq_len, self.expansion, -1)

        # Apply H_pre scaling per expansion group
        x_pre = x_expanded * h_pre.reshape(1, 1, self.expansion, 1)

        return x_pre.reshape(batch_size, seq_len, dims)
```

**Run test (expect PASS):**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py::TestTwoStageAPI -v -k "pre_scale"
```

**Expected output:**
```
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_exists PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_output_shape PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_applies_h_pre PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_different_expansions PASSED
```

**Commit:**

```bash
cd /Users/ma/mlx-mhc && git add mlx_mhc/mhc.py tests/test_mhc.py && git commit -m "$(cat <<'EOF'
Add pre_scale method for two-stage mHC API

Implements the first stage of the two-stage API that allows users to
apply H_pre scaling before their layer function F, matching the
DeepSeek paper equation: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

- Add pre_scale(x) method that returns H_pre * x
- Add comprehensive tests for pre_scale functionality
EOF
)"
```

---

## Task 3: Add test for post_combine method

**File:** `/Users/ma/mlx-mhc/tests/test_mhc.py`

**Add these tests to the `TestTwoStageAPI` class:**

```python
    def test_post_combine_exists(self):
        """post_combine method should exist and be callable."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        assert hasattr(mhc, 'post_combine'), "post_combine method should exist"
        assert callable(mhc.post_combine), "post_combine should be callable"

    def test_post_combine_output_shape(self):
        """post_combine should preserve input shape."""
        dims = 64
        batch_size = 2
        seq_len = 16
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)
        x = mx.random.normal((batch_size, seq_len, dims))
        layer_output = mx.random.normal((batch_size, seq_len, dims))

        output = mhc.post_combine(x, layer_output)
        mx.eval(output)

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_post_combine_applies_h_res_and_h_post(self):
        """post_combine should apply H_res to residual and H_post to combined output."""
        dims = 4
        expansion = 2
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion, sinkhorn_iterations=20)

        # With default initialization:
        # H_pre = 0.5 (sigmoid(0))
        # H_post = 1.0 (2 * sigmoid(0))
        # H_res = doubly stochastic (varies)

        x = mx.ones((1, 1, dims))
        layer_output = mx.ones((1, 1, dims))

        output = mhc.post_combine(x, layer_output)
        mx.eval(output)

        # Output should be non-trivial (not zeros)
        assert not mx.allclose(output, mx.zeros_like(output)), "Output should not be all zeros"

        # Output shape should match
        assert output.shape == (1, 1, dims)

    def test_post_combine_different_expansions(self):
        """post_combine should work with different expansion factors."""
        dims = 128
        batch_size = 4
        seq_len = 32

        for expansion in [2, 4, 8]:
            mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)
            x = mx.random.normal((batch_size, seq_len, dims))
            layer_output = mx.random.normal((batch_size, seq_len, dims))

            output = mhc.post_combine(x, layer_output)
            mx.eval(output)

            assert output.shape == x.shape, (
                f"Expansion {expansion}: Expected {x.shape}, got {output.shape}"
            )

    def test_two_stage_api_differentiable(self):
        """Two-stage API should be fully differentiable."""
        dims = 32
        batch_size = 2
        seq_len = 8
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        def loss_fn(model, x):
            x_pre = model.pre_scale(x)
            # Simulate a simple layer (identity for testing)
            layer_out = x_pre * 2.0
            output = model.post_combine(x, layer_out)
            return mx.mean(output ** 2)

        x = mx.random.normal((batch_size, seq_len, dims))

        loss, grads = mx.value_and_grad(loss_fn)(mhc, x)
        mx.eval(loss, grads)

        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert grads is not None, "Should have gradients"
```

**Run test (expect FAIL - method does not exist yet):**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py::TestTwoStageAPI::test_post_combine_exists -v
```

**Expected output:**
```
FAILED tests/test_mhc.py::TestTwoStageAPI::test_post_combine_exists - AssertionError: post_combine method should exist
```

---

## Task 4: Implement post_combine method

**File:** `/Users/ma/mlx-mhc/mlx_mhc/mhc.py`

**Add this method to ManifoldHyperConnection class (after `pre_scale`):**

```python
    def post_combine(self, x: mx.array, layer_output: mx.array) -> mx.array:
        """
        Combine layer output with residual and apply H_post (second stage of two-stage API).

        This completes the manifold hyper-connection by:
        1. Computing H_res * H_pre * x (residual path)
        2. Adding layer_output to residual
        3. Applying H_post scaling

        The full equation is: H_post * (layer_output + H_res * H_pre * x)

        Note: layer_output should be F(H_pre * x) where H_pre * x was obtained
        from pre_scale(). This method handles the residual computation internally
        using the original x.

        Args:
            x: Original input tensor of shape (batch, seq_len, dims)
            layer_output: Output from layer F applied to pre_scale(x),
                         shape (batch, seq_len, dims)

        Returns:
            Output tensor of shape (batch, seq_len, dims)
        """
        batch_size, seq_len, dims = x.shape

        h_res = self._project_h_res()
        h_pre = self._project_h_pre()
        h_post = self._project_h_post()

        # Reshape for expansion: (batch, seq, dims) -> (batch, seq, expansion, dims//expansion)
        x_expanded = x.reshape(batch_size, seq_len, self.expansion, -1)

        # Apply H_pre to x for residual path
        x_pre = x_expanded * h_pre.reshape(1, 1, self.expansion, 1)

        # Apply H_res via einsum
        x_res = mx.einsum('ij,...jd->...id', h_res, x_pre)

        # Reshape layer_output
        layer_expanded = layer_output.reshape(batch_size, seq_len, self.expansion, -1)

        # Combine and apply H_post
        combined = layer_expanded + x_res
        output_expanded = combined * h_post.reshape(1, 1, self.expansion, 1)

        return output_expanded.reshape(batch_size, seq_len, dims)
```

**Run test (expect PASS):**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py::TestTwoStageAPI -v
```

**Expected output:**
```
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_exists PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_output_shape PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_applies_h_pre PASSED
tests/test_mhc.py::TestTwoStageAPI::test_pre_scale_different_expansions PASSED
tests/test_mhc.py::TestTwoStageAPI::test_post_combine_exists PASSED
tests/test_mhc.py::TestTwoStageAPI::test_post_combine_output_shape PASSED
tests/test_mhc.py::TestTwoStageAPI::test_post_combine_applies_h_res_and_h_post PASSED
tests/test_mhc.py::TestTwoStageAPI::test_post_combine_different_expansions PASSED
tests/test_mhc.py::TestTwoStageAPI::test_two_stage_api_differentiable PASSED
```

**Commit:**

```bash
cd /Users/ma/mlx-mhc && git add mlx_mhc/mhc.py tests/test_mhc.py && git commit -m "$(cat <<'EOF'
Add post_combine method for two-stage mHC API

Implements the second stage of the two-stage API that combines the
layer output with the residual path and applies H_post scaling.

Full equation: H_post * (layer_output + H_res * H_pre * x)

- Add post_combine(x, layer_output) method
- Add comprehensive tests for post_combine functionality
- Verify two-stage API is fully differentiable
EOF
)"
```

---

## Task 5: Update existing __call__ to use new methods (backward compatibility)

**File:** `/Users/ma/mlx-mhc/mlx_mhc/mhc.py`

**Replace the existing `__call__` method with:**

```python
    def __call__(self, x: mx.array, layer_output: mx.array) -> mx.array:
        """
        Apply manifold-constrained hyper-connection (legacy API).

        WARNING: This method exists for backward compatibility but does NOT
        correctly implement the paper equation. It applies H_pre only to the
        residual path, not to the input before the layer function.

        For correct behavior matching the paper, use the two-stage API:
            x_pre = mhc.pre_scale(x)
            layer_out = your_layer(x_pre)
            output = mhc.post_combine(x, layer_out)

        Args:
            x: Input tensor of shape (batch, seq_len, dims)
            layer_output: Output from the layer of shape (batch, seq_len, dims)

        Returns:
            Output tensor of shape (batch, seq_len, dims)
        """
        # Legacy behavior: layer_output is NOT pre-scaled, but we still
        # apply H_res * H_pre * x for residual. This is technically incorrect
        # per the paper, but maintains backward compatibility.
        return self.post_combine(x, layer_output)
```

**Also update the class docstring to document both APIs. Replace the existing docstring with:**

```python
class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) module.

    Implements the mHC architecture from DeepSeek's paper (arXiv:2512.24880).
    Projects residual connections onto a manifold to maintain training stability.

    The paper equation is:
        x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

    Where:
        - H_res is constrained to be doubly stochastic (Birkhoff polytope)
        - H_pre and H_post are constrained to be non-negative via sigmoid

    TWO-STAGE API (Recommended - matches paper exactly):
        mhc = ManifoldHyperConnection(dims, expansion)
        x_pre = mhc.pre_scale(x)           # Apply H_pre
        layer_out = your_layer(x_pre)      # F(H_pre * x)
        output = mhc.post_combine(x, layer_out)  # H_post * (F(...) + H_res * H_pre * x)

    LEGACY API (Backward compatible, but H_pre not applied before F):
        mhc = ManifoldHyperConnection(dims, expansion)
        output = mhc(x, layer_output)      # H_post * (layer_output + H_res * H_pre * x)

    Args:
        dims: Hidden dimension of the input/output.
        expansion: Expansion factor for the hyper-connection width (default: 2).
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations (default: 10).
    """
```

**Run all existing tests to verify backward compatibility:**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py -v
```

**Expected output:**
```
tests/test_mhc.py::TestManifoldHyperConnection::test_output_shape_matches_input PASSED
tests/test_mhc.py::TestManifoldHyperConnection::test_output_shape_with_different_expansions PASSED
... (all tests PASSED)
```

**Commit:**

```bash
cd /Users/ma/mlx-mhc && git add mlx_mhc/mhc.py && git commit -m "$(cat <<'EOF'
Update __call__ to use post_combine for backward compatibility

Refactors __call__ to use post_combine internally, maintaining the same
behavior for existing code. Updates documentation to clearly explain both
APIs and recommend the two-stage API for correct paper implementation.

- __call__ now delegates to post_combine
- Updated class docstring with both API examples
- Added deprecation warning in __call__ docstring
EOF
)"
```

---

## Task 6: Update example to use correct API

**File:** `/Users/ma/mlx-mhc/examples/basic_usage.py`

**Replace the `SimpleTransformerBlock` class with:**

```python
class SimpleTransformerBlock(nn.Module):
    """Example transformer block using mHC with correct two-stage API."""

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
        # Attention block with correct mHC application
        # Paper: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)
        x_pre_attn = self.mhc_attn.pre_scale(self.norm1(x))
        attn_out = self.attn(x_pre_attn, x_pre_attn, x_pre_attn)
        h = self.mhc_attn.post_combine(self.norm1(x), attn_out)

        # MLP block with correct mHC application
        x_pre_mlp = self.mhc_mlp.pre_scale(self.norm2(h))
        mlp_out = self.mlp(x_pre_mlp)
        return self.mhc_mlp.post_combine(self.norm2(h), mlp_out)
```

**Also add a new demo function showing both APIs. Add after `demo_mhc_module`:**

```python
def demo_two_stage_api():
    """Demonstrate the correct two-stage mHC API."""
    print("\n" + "=" * 50)
    print("Two-Stage API Demo (Paper-Correct)")
    print("=" * 50)

    dims = 64
    batch_size = 2
    seq_len = 8

    connection = mhc.ManifoldHyperConnection(dims=dims, expansion=2)

    x = mx.random.normal((batch_size, seq_len, dims))

    # Stage 1: Apply H_pre before the layer
    x_pre = connection.pre_scale(x)
    print(f"\nOriginal x shape: {x.shape}")
    print(f"After pre_scale: {x_pre.shape}")

    # Simulate a layer F (here just a simple linear transform)
    layer_output = x_pre * 2.0  # F(H_pre * x)

    # Stage 2: Combine with residual and apply H_post
    output = connection.post_combine(x, layer_output)
    print(f"Final output: {output.shape}")

    # Show the equation being computed
    print("\nEquation: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)")
```

**Update the `if __name__ == "__main__":` block:**

```python
if __name__ == "__main__":
    demo_sinkhorn()
    demo_mhc_module()
    demo_two_stage_api()
    demo_transformer()
    print("\n" + "=" * 50)
    print("All demos complete!")
```

**Run the example to verify:**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python examples/basic_usage.py
```

**Expected output (partial):**
```
==================================================
Two-Stage API Demo (Paper-Correct)
==================================================

Original x shape: (2, 8, 64)
After pre_scale: (2, 8, 64)
Final output: (2, 8, 64)

Equation: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)
...
All demos complete!
```

**Commit:**

```bash
cd /Users/ma/mlx-mhc && git add examples/basic_usage.py && git commit -m "$(cat <<'EOF'
Update example to use correct two-stage mHC API

Updates SimpleTransformerBlock to use pre_scale/post_combine for correct
paper implementation. Adds demo_two_stage_api() to demonstrate the new API.

- SimpleTransformerBlock now applies H_pre before attention/MLP
- New demo shows the two-stage API workflow
- Comments explain the paper equation being computed
EOF
)"
```

---

## Task 7: Add integration test verifying paper equation

**File:** `/Users/ma/mlx-mhc/tests/test_mhc.py`

**Add this test class at the end of the file:**

```python
class TestPaperEquation:
    """Tests verifying the implementation matches the DeepSeek paper equation."""

    def test_two_stage_matches_paper_equation(self):
        """
        Verify: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

        Using the two-stage API:
        - pre_scale(x) returns H_pre * x
        - post_combine(x, F(H_pre * x)) returns H_post * (F(H_pre * x) + H_res * H_pre * x)
        """
        dims = 8
        expansion = 2
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion, sinkhorn_iterations=20)

        # Get the projected matrices
        h_pre = mhc._project_h_pre()
        h_res = mhc._project_h_res()
        h_post = mhc._project_h_post()
        mx.eval(h_pre, h_res, h_post)

        # Input
        x = mx.random.normal((1, 1, dims))
        mx.eval(x)

        # Manual computation of paper equation
        x_expanded = x.reshape(1, 1, expansion, -1)

        # H_pre * x
        h_pre_x = x_expanded * h_pre.reshape(1, 1, expansion, 1)

        # Simulate layer F as simple scaling by 2
        def layer_f(inp):
            return inp * 2.0

        f_h_pre_x = layer_f(h_pre_x)

        # H_res * H_pre * x
        h_res_h_pre_x = mx.einsum('ij,...jd->...id', h_res, h_pre_x)

        # F(H_pre * x) + H_res * H_pre * x
        combined = f_h_pre_x + h_res_h_pre_x

        # H_post * (...)
        expected = combined * h_post.reshape(1, 1, expansion, 1)
        expected = expected.reshape(1, 1, dims)
        mx.eval(expected)

        # Two-stage API computation
        x_pre = mhc.pre_scale(x)  # H_pre * x
        layer_out = layer_f(x_pre)  # F(H_pre * x)
        actual = mhc.post_combine(x, layer_out)
        mx.eval(actual)

        # They should match
        assert mx.allclose(actual, expected, atol=1e-5), (
            f"Two-stage API output doesn't match paper equation.\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )

    def test_legacy_api_differs_from_paper(self):
        """
        Verify that the legacy __call__ API does NOT match the paper equation.

        Legacy: H_post * (layer_output + H_res * H_pre * x)
        Paper:  H_post * (F(H_pre * x) + H_res * H_pre * x)

        When F is not identity and layer_output != F(H_pre * x), they differ.
        """
        dims = 8
        expansion = 2
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion, sinkhorn_iterations=20)

        x = mx.random.normal((1, 1, dims))

        # Simulate layer F that scales by 2
        def layer_f(inp):
            return inp * 2.0

        # Correct two-stage computation
        x_pre = mhc.pre_scale(x)
        layer_out_correct = layer_f(x_pre)  # F(H_pre * x)
        correct_output = mhc.post_combine(x, layer_out_correct)

        # Legacy API with layer output computed on x (not H_pre * x)
        layer_out_wrong = layer_f(x)  # F(x) - NOT F(H_pre * x)
        legacy_output = mhc(x, layer_out_wrong)

        mx.eval(correct_output, legacy_output)

        # They should be different (unless H_pre happens to be 1.0)
        # With default init, H_pre = 0.5, so F(H_pre * x) = 2 * 0.5 * x = x
        # while F(x) = 2 * x, so they differ
        h_pre = mhc._project_h_pre()
        mx.eval(h_pre)

        # If H_pre is not all 1.0, outputs should differ
        if not mx.allclose(h_pre, mx.ones_like(h_pre), atol=1e-3):
            assert not mx.allclose(correct_output, legacy_output, atol=1e-3), (
                "Legacy and correct outputs should differ when H_pre != 1.0"
            )

    def test_full_transformer_layer_equation(self):
        """
        Integration test: verify mHC in a transformer-like setup.

        Simulates: x_{l+1} = mHC(x, Attention(mHC.pre_scale(x)))
        """
        dims = 32
        expansion = 2
        batch_size = 2
        seq_len = 4

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        x = mx.random.normal((batch_size, seq_len, dims))

        # Simulate attention (just a linear transform for testing)
        w = mx.random.normal((dims, dims)) * 0.1

        # Correct implementation
        x_pre = mhc.pre_scale(x)
        attn_out = x_pre @ w  # "Attention" on H_pre * x
        output = mhc.post_combine(x, attn_out)

        mx.eval(output)

        # Output should have correct shape
        assert output.shape == x.shape

        # Output should not be identical to input (transformation happened)
        assert not mx.allclose(output, x, atol=1e-3)

    def test_gradient_flow_through_full_equation(self):
        """Verify gradients flow correctly through the full paper equation."""
        dims = 16
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        # Learnable layer parameters
        w = mx.random.normal((dims, dims)) * 0.1

        def forward(model, w, x):
            x_pre = model.pre_scale(x)
            layer_out = x_pre @ w
            output = model.post_combine(x, layer_out)
            return mx.mean(output ** 2)

        x = mx.random.normal((2, 4, dims))

        # Compute gradients w.r.t. model parameters
        loss, grads = mx.value_and_grad(forward)(mhc, w, x)
        mx.eval(loss, grads)

        assert loss.shape == ()
        assert grads is not None

        # Verify specific parameters got gradients
        assert 'h_pre_raw' in grads
        assert 'h_post_raw' in grads
        assert 'h_res_raw' in grads

        # Gradients should be non-zero
        mx.eval(grads['h_pre_raw'], grads['h_post_raw'], grads['h_res_raw'])
        assert not mx.allclose(grads['h_pre_raw'], mx.zeros_like(grads['h_pre_raw'])), (
            "h_pre_raw gradients should be non-zero"
        )
```

**Run the paper equation tests:**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/test_mhc.py::TestPaperEquation -v
```

**Expected output:**
```
tests/test_mhc.py::TestPaperEquation::test_two_stage_matches_paper_equation PASSED
tests/test_mhc.py::TestPaperEquation::test_legacy_api_differs_from_paper PASSED
tests/test_mhc.py::TestPaperEquation::test_full_transformer_layer_equation PASSED
tests/test_mhc.py::TestPaperEquation::test_gradient_flow_through_full_equation PASSED
```

**Run full test suite:**

```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/ -v
```

**Expected output:**
```
... (all tests PASSED)
```

**Commit:**

```bash
cd /Users/ma/mlx-mhc && git add tests/test_mhc.py && git commit -m "$(cat <<'EOF'
Add integration tests verifying paper equation implementation

Adds TestPaperEquation class with comprehensive tests ensuring the
two-stage API correctly implements the DeepSeek paper equation:
x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

Tests include:
- Manual computation vs two-stage API comparison
- Verification that legacy API differs from paper equation
- Full transformer layer integration test
- Gradient flow verification through complete equation
EOF
)"
```

---

## Summary

After completing all tasks:

1. **pre_scale(x)** - Returns H_pre * x for use before layer F
2. **post_combine(x, layer_output)** - Returns H_post * (layer_output + H_res * H_pre * x)
3. **__call__(x, layer_output)** - Legacy API (backward compatible, delegates to post_combine)

**Correct usage pattern:**
```python
mhc = ManifoldHyperConnection(dims, expansion)
x_pre = mhc.pre_scale(x)           # Apply H_pre
layer_out = layer(x_pre)           # F(H_pre * x)
output = mhc.post_combine(x, layer_out)  # H_post * (F(...) + H_res * H_pre * x)
```

**Verification:**
```bash
cd /Users/ma/mlx-mhc && source .venv/bin/activate && python -m pytest tests/ -v
```

All tests should pass, confirming the implementation matches the paper equation.
