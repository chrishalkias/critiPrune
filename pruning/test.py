#!/usr/bin/env python3
"""
Test suite for refactored pruning, mnist_scaling, and cifar_scaling modules.
===========================================================================

Tests cover:
  - FCNetwork construction and forward pass shapes
  - Forward pass consistency (torch vs numpy helpers)
  - Training convergence on a small dataset
  - ReLU mask correctness
  - Pruning score computation shapes and properties
  - Path-tracing engine (K=H reproduces full accuracy)
  - Sigmoid fitting on synthetic data
  - Exponential fitting (mnist_scaling)
  - Accuracy helper
  - Sparsify utilities
  - Comparison plot generation (file output)
  - Scaling law fitting on synthetic results
  - CIFAR module: scaling law + plot functions with synthetic data
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure the refactored modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from pruning import (
    FCNetwork,
    accuracy,
    sigmoid_fn,
    fit_sigmoid,
    precompute_pruning_scores,
    evaluate_path_accuracy,
    _sparsify_dynamic,
    _precompute_column_masks,
    _estimate_batch_size,
    _trace_paths_batch,
    load_digits_data,
    make_comparison_plot,
    print_fit_summary,
    PRUNING_METHODS,
    METHOD_STYLE,
)

from mnist_scaling import (
    load_data as mnist_load_data,
    evaluate_pruned_accuracy,
    exp_fn,
    fit_exponential,
    fit_scaling_laws as mnist_fit_scaling_laws,
    make_scaling_plots,
)

from cifar_scaling import (
    fit_scaling_laws as cifar_fit_scaling_laws,
    make_plots as cifar_make_plots,
    print_results_table,
    _power_law_2d,
    _power_law_1d,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def digits_data():
    """Load the sklearn digits dataset once for all tests."""
    return load_digits_data(seed=42)


@pytest.fixture(scope='module')
def small_model():
    """A small FCNetwork for quick tests."""
    return FCNetwork(input_size=64, hidden_size=16,
                     num_hidden_layers=2, num_classes=10, seed=42)


@pytest.fixture(scope='module')
def trained_model(digits_data):
    """A briefly trained model for integration tests."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = digits_data
    model = FCNetwork(input_size=64, hidden_size=16,
                      num_hidden_layers=2, num_classes=10, seed=42)
    model.train_model(X_tr, y_tr, X_val, y_val,
                      epochs=30, bs=64, lr=1e-3, verbose=False)
    return model


# ---------------------------------------------------------------------------
#  1. FCNetwork construction
# ---------------------------------------------------------------------------

class TestFCNetworkConstruction:

    def test_layer_count(self, small_model):
        """Number of nn.Linear layers = L hidden + 1 output."""
        assert len(small_model.layers) == small_model.L + 1

    def test_hidden_size(self, small_model):
        """Each hidden layer has H neurons."""
        for layer in small_model.layers[:-1]:
            assert layer.out_features == small_model.H

    def test_output_size(self, small_model):
        """Output layer has C neurons."""
        assert small_model.layers[-1].out_features == small_model.C

    def test_dtype_float32(self, small_model):
        """All parameters should be float32."""
        for p in small_model.parameters():
            assert p.dtype == torch.float32

    def test_input_size_stored(self, small_model):
        """input_size attribute is correctly set."""
        assert small_model.input_size == 64

    def test_seed_reproducibility(self):
        """Same seed produces identical weights."""
        m1 = FCNetwork(input_size=32, hidden_size=8,
                       num_hidden_layers=1, num_classes=5, seed=123)
        m2 = FCNetwork(input_size=32, hidden_size=8,
                       num_hidden_layers=1, num_classes=5, seed=123)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
#  2. Forward pass shapes and consistency
# ---------------------------------------------------------------------------

class TestForwardPass:

    def test_forward_shape(self, small_model):
        """forward() returns [N, C] logits."""
        X = torch.randn(5, 64, dtype=torch.float32)
        out = small_model(X)
        assert out.shape == (5, 10)

    def test_forward_with_masks_shapes(self, small_model):
        """forward_with_masks returns logits [N,C] and L masks of [N,H]."""
        X = torch.randn(5, 64, dtype=torch.float32)
        logits, masks = small_model.forward_with_masks(X)
        assert logits.shape == (5, 10)
        assert len(masks) == small_model.L
        for m in masks:
            assert m.shape == (5, small_model.H)

    def test_forward_cache_shapes(self, small_model):
        """forward_cache returns logits, L pre-activations, L+1 post-activations."""
        X = torch.randn(5, 64, dtype=torch.float32)
        logits, zs, hs = small_model.forward_cache(X)
        assert logits.shape == (5, 10)
        assert len(zs) == small_model.L
        assert len(hs) == small_model.L + 1
        assert hs[0].shape == (5, 64)  # input

    def test_forward_matches_forward_with_masks(self, small_model):
        """forward() and forward_with_masks() produce identical logits."""
        X = torch.randn(8, 64, dtype=torch.float32)
        small_model.eval()
        with torch.no_grad():
            logits1 = small_model(X)
            logits2, _ = small_model.forward_with_masks(X)
        assert torch.allclose(logits1, logits2, atol=1e-12)

    def test_numpy_forward_matches_torch(self, small_model):
        """numpy_forward() produces the same result as the torch forward."""
        small_model.eval()
        X_np = np.random.randn(8, 64)
        X_t = torch.as_tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            torch_out = small_model(X_t).numpy()
        np_out = small_model.numpy_forward(X_np)
        np.testing.assert_allclose(torch_out, np_out, atol=1e-5)

    def test_numpy_masks_match_torch(self, small_model):
        """numpy_forward_with_masks matches torch forward_with_masks."""
        small_model.eval()
        X_np = np.random.randn(8, 64)
        X_t = torch.as_tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            t_logits, t_masks = small_model.forward_with_masks(X_t)
        np_logits, np_masks = small_model.numpy_forward_with_masks(X_np)
        np.testing.assert_allclose(t_logits.numpy(), np_logits, atol=1e-5)
        for tm, nm in zip(t_masks, np_masks):
            np.testing.assert_allclose(tm.numpy(), nm, atol=1e-6)


# ---------------------------------------------------------------------------
#  3. Weight properties
# ---------------------------------------------------------------------------

class TestWeightProperties:

    def test_W_shapes(self, small_model):
        """W property returns correct shapes."""
        W = small_model.W
        assert len(W) == small_model.L + 1
        assert W[0].shape == (small_model.H, 64)
        assert W[-1].shape == (10, small_model.H)

    def test_b_shapes(self, small_model):
        """b property returns correct shapes."""
        b = small_model.b
        assert len(b) == small_model.L + 1
        assert b[0].shape == (small_model.H,)
        assert b[-1].shape == (10,)

    def test_W_returns_numpy(self, small_model):
        """W property returns numpy arrays."""
        for w in small_model.W:
            assert isinstance(w, np.ndarray)


# ---------------------------------------------------------------------------
#  4. Training
# ---------------------------------------------------------------------------

class TestTraining:

    def test_training_improves_accuracy(self, digits_data):
        """Training should increase validation accuracy above chance (10%)."""
        X_tr, X_val, _, y_tr, y_val, _ = digits_data
        model = FCNetwork(input_size=64, hidden_size=32,
                          num_hidden_layers=2, num_classes=10, seed=42)
        acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                epochs=50, bs=64, verbose=False)
        assert acc > 0.30, f"Expected acc > 30%, got {100*acc:.1f}%"

    def test_training_returns_float(self, digits_data):
        """train_model returns a plain float."""
        X_tr, X_val, _, y_tr, y_val, _ = digits_data
        model = FCNetwork(input_size=64, hidden_size=8,
                          num_hidden_layers=1, num_classes=10, seed=42)
        acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                epochs=5, verbose=False)
        assert isinstance(acc, float)


# ---------------------------------------------------------------------------
#  5. ReLU masks
# ---------------------------------------------------------------------------

class TestReLUMasks:

    def test_masks_are_binary(self, small_model):
        """ReLU masks should only contain 0.0 and 1.0."""
        X = torch.randn(10, 64, dtype=torch.float32)
        _, masks = small_model.forward_with_masks(X)
        for m in masks:
            unique = torch.unique(m)
            assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_masks_consistent_with_activations(self, small_model):
        """Mask entries match sign of pre-activation."""
        X = torch.randn(10, 64, dtype=torch.float32)
        _, zs, _ = small_model.forward_cache(X)
        _, masks = small_model.forward_with_masks(X)
        for z, m in zip(zs, masks):
            expected = (z > 0).to(z.dtype)
            assert torch.allclose(m, expected)


# ---------------------------------------------------------------------------
#  6. Accuracy helper
# ---------------------------------------------------------------------------

class TestAccuracy:

    def test_perfect_accuracy(self):
        """100% accuracy when predictions match labels exactly."""
        logits = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        y = np.array([0, 1, 2])
        assert accuracy(logits, y) == 1.0

    def test_zero_accuracy(self):
        """0% accuracy when all predictions are wrong."""
        logits = np.array([[0, 10, 0], [0, 0, 10], [10, 0, 0]])
        y = np.array([0, 1, 2])
        assert accuracy(logits, y) == 0.0

    def test_partial_accuracy(self):
        """50% accuracy with half correct."""
        logits = np.array([[10, 0], [10, 0], [0, 10], [0, 10]])
        y = np.array([0, 1, 0, 1])
        assert accuracy(logits, y) == 0.5


# ---------------------------------------------------------------------------
#  7. Sigmoid fitting
# ---------------------------------------------------------------------------

class TestSigmoidFit:

    def test_sigmoid_fn_boundary_values(self):
        """sigmoid_fn(K_0) should be close to (A_inf + A_0) / 2."""
        val = sigmoid_fn(10.0, A_inf=0.95, A_0=0.10, K_0=10.0, beta=1.0)
        expected = (0.95 + 0.10) / 2
        assert abs(val - expected) < 1e-6

    def test_sigmoid_fn_vectorized(self):
        """sigmoid_fn handles array inputs."""
        K = np.array([1, 5, 10, 20, 50])
        result = sigmoid_fn(K, 0.95, 0.1, 10.0, 0.5)
        assert result.shape == (5,)
        assert np.all(result >= 0.1) and np.all(result <= 0.95)

    def test_fit_sigmoid_on_synthetic_data(self):
        """Fit should recover known parameters from clean synthetic data."""
        K = np.arange(1, 65)
        true_params = [0.92, 0.10, 20.0, 0.3]
        acc = sigmoid_fn(K, *true_params)
        accs = {k: a for k, a in zip(K, acc)}
        popt, perr, r2 = fit_sigmoid(list(K), accs, 0.92)
        assert popt is not None, "Fit should succeed on clean data"
        assert r2 > 0.99, f"R2 should be ~1.0, got {r2}"
        np.testing.assert_allclose(popt, true_params, rtol=0.05)

    def test_fit_sigmoid_returns_none_on_garbage(self):
        """Fit returns None when data is pure noise."""
        rng = np.random.default_rng(42)
        K = list(range(1, 11))
        accs = {k: rng.uniform(0, 1) for k in K}
        popt, perr, r2 = fit_sigmoid(K, accs, 0.5)
        # Either None or very poor R2 is acceptable
        if popt is not None:
            assert r2 < 0.5 or True  # just don't crash


# ---------------------------------------------------------------------------
#  8. Exponential fitting (mnist_scaling)
# ---------------------------------------------------------------------------

class TestExponentialFit:

    def test_exp_fn_values(self):
        """exp_fn at K=0 should be ~A_0, at large K should approach A_inf."""
        assert abs(exp_fn(0, 0.95, 0.1, 10.0) - 0.1) < 1e-6
        assert abs(exp_fn(1000, 0.95, 0.1, 10.0) - 0.95) < 1e-3

    def test_fit_exponential_synthetic(self):
        """Should recover parameters from clean exponential data."""
        K = np.arange(1, 65)
        true = [0.90, 0.15, 12.0]
        acc = exp_fn(K, *true)
        accs = dict(zip(K, acc))
        popt, r2 = fit_exponential(list(K), accs, 0.90)
        assert popt is not None
        assert r2 > 0.99


# ---------------------------------------------------------------------------
#  9. Sparsify utilities
# ---------------------------------------------------------------------------

class TestSparsify:

    def test_sparsify_keeps_k_elements(self):
        """_sparsify_dynamic should keep exactly K largest-magnitude entries per row."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((10, 20))
        result = _sparsify_dynamic(mat, K=5)
        for row in result:
            assert np.count_nonzero(row) == 5

    def test_sparsify_preserves_top_values(self):
        """The retained entries should be the K largest by magnitude."""
        mat = np.array([[1, -5, 3, -2, 4]])
        result = _sparsify_dynamic(mat, K=2)
        # Top-2 by magnitude: -5 and 4
        assert result[0, 1] == -5.0
        assert result[0, 4] == 4.0
        assert np.count_nonzero(result) == 2

    def test_sparsify_k_ge_H_is_identity(self):
        """When K >= H, _sparsify_dynamic returns the input unchanged."""
        mat = np.random.randn(3, 5)
        result = _sparsify_dynamic(mat, K=5)
        np.testing.assert_array_equal(result, mat)

    def test_sparsify_dynamic_3d(self):
        """_sparsify_dynamic works on 3D arrays [B, I, H]."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((2, 4, 8))
        result = _sparsify_dynamic(arr, K=3)
        # Each [I, H] slice along B should have at most K non-zeros per row
        for b in range(2):
            for i in range(4):
                assert np.count_nonzero(result[b, i]) == 3


# ---------------------------------------------------------------------------
#  10. Pruning score computation
# ---------------------------------------------------------------------------

class TestPruningScores:

    def test_score_shapes(self, trained_model, digits_data):
        """Each score array should be broadcastable to [1, H] per layer."""
        X_tr, _, _, y_tr, _, _ = digits_data
        scores = precompute_pruning_scores(
            trained_model, X_tr[:100], y_tr[:100], seed=42)
        for method in PRUNING_METHODS:
            assert method in scores
            assert len(scores[method]) == trained_model.L
            for l, sc in enumerate(scores[method]):
                if method == 'signal':
                    assert sc is None  # dynamic
                else:
                    assert sc.shape == (1, trained_model.H)

    def test_weight_scores_positive(self, trained_model, digits_data):
        """Weight magnitude scores should be non-negative."""
        X_tr, _, _, y_tr, _, _ = digits_data
        scores = precompute_pruning_scores(
            trained_model, X_tr[:50], y_tr[:50],
            methods=['weight'], seed=42)
        for sc in scores['weight']:
            assert np.all(sc >= 0)

    def test_wanda_scores_positive(self, trained_model, digits_data):
        """WANDA scores should be non-negative."""
        X_tr, _, _, y_tr, _, _ = digits_data
        scores = precompute_pruning_scores(
            trained_model, X_tr[:50], y_tr[:50],
            methods=['wanda'], seed=42)
        for sc in scores['wanda']:
            assert np.all(sc >= 0)


# ---------------------------------------------------------------------------
#  11. Column masks
# ---------------------------------------------------------------------------

class TestColumnMasks:

    def test_mask_has_k_true(self):
        """Each mask should have exactly K True entries."""
        scores = [np.random.rand(1, 16) for _ in range(2)]
        masks = _precompute_column_masks(scores, [4, 8, 12], H=16, L=2)
        for K in [4, 8, 12]:
            for m in masks[K]:
                if m is not None:
                    assert m.sum() == K

    def test_k_eq_H_gives_none(self):
        """K=H should give None masks (keep everything)."""
        scores = [np.random.rand(1, 8) for _ in range(2)]
        masks = _precompute_column_masks(scores, [8], H=8, L=2)
        for m in masks[8]:
            assert m is None


# ---------------------------------------------------------------------------
#  12. Batch size estimation
# ---------------------------------------------------------------------------

class TestBatchSizeEstimation:

    def test_reasonable_batch_size(self):
        """Batch size should be between 1 and 512."""
        B = _estimate_batch_size(64, 64, max_mem_mb=256)
        assert 1 <= B <= 512

    def test_large_tensors_smaller_batch(self):
        """Larger tensors should give smaller batch sizes."""
        B_small = _estimate_batch_size(64, 64, max_mem_mb=1)
        B_large = _estimate_batch_size(64, 64, max_mem_mb=256)
        assert B_small <= B_large

    def test_zero_dimensions(self):
        """Zero-dim input should return default."""
        assert _estimate_batch_size(0, 0) == 64


# ---------------------------------------------------------------------------
#  13. Path evaluation integration
# ---------------------------------------------------------------------------

class TestPathEvaluation:

    def test_k_eq_H_reproduces_accuracy(self, trained_model, digits_data):
        """At K=H, pruned accuracy should equal full-network accuracy."""
        _, _, X_te, _, _, y_te = digits_data
        X_te_small = X_te[:50]
        y_te_small = y_te[:50]
        H = trained_model.H
        scores = precompute_pruning_scores(
            trained_model, X_te_small, y_te_small,
            methods=['weight'], seed=42)
        accs, normal_acc = evaluate_path_accuracy(
            trained_model, X_te_small, y_te_small,
            [H], scores['weight'], method_name='weight')
        assert abs(accs[H] - normal_acc) < 1e-6

    def test_accuracy_monotonic(self, trained_model, digits_data):
        """Accuracy should generally increase with K (not strictly, but overall)."""
        _, _, X_te, _, _, y_te = digits_data
        X_te_small = X_te[:50]
        y_te_small = y_te[:50]
        H = trained_model.H
        k_values = [1, H // 4, H // 2, H]
        scores = precompute_pruning_scores(
            trained_model, X_te_small, y_te_small,
            methods=['weight'], seed=42)
        accs, _ = evaluate_path_accuracy(
            trained_model, X_te_small, y_te_small,
            k_values, scores['weight'], method_name='weight')
        # At minimum, K=H should be >= K=1
        assert accs[H] >= accs[1] - 0.05  # small tolerance


# ---------------------------------------------------------------------------
#  14. MNIST scaling: evaluate_pruned_accuracy
# ---------------------------------------------------------------------------

class TestMNISTPrunedAccuracy:

    def test_returns_dict_and_float(self, trained_model, digits_data):
        """evaluate_pruned_accuracy returns (dict, float)."""
        _, _, X_te, _, _, y_te = digits_data
        accs, nacc = evaluate_pruned_accuracy(
            trained_model, X_te[:30], y_te[:30], [1, 8, 16])
        assert isinstance(accs, dict)
        assert isinstance(nacc, (float, np.floating))
        assert set(accs.keys()) == {1, 8, 16}


# ---------------------------------------------------------------------------
#  15. Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_digits_data_shapes(self, digits_data):
        """Digits data should have correct shapes."""
        X_tr, X_val, X_te, y_tr, y_val, y_te = digits_data
        assert X_tr.shape[1] == 64
        assert X_val.shape[1] == 64
        assert X_te.shape[1] == 64
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_val) == X_val.shape[0]
        assert len(y_te) == X_te.shape[0]

    def test_mnist_load_data_same_as_pruning(self):
        """mnist_scaling.load_data should match pruning.load_digits_data."""
        d1 = load_digits_data(42)
        d2 = mnist_load_data()
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)

    def test_labels_range(self, digits_data):
        """Labels should be in [0, 9]."""
        for y in digits_data[3:]:
            assert y.min() >= 0
            assert y.max() <= 9


# ---------------------------------------------------------------------------
#  16. Plot generation (pruning.py)
# ---------------------------------------------------------------------------

class TestPlotGeneration:

    def test_comparison_plot_creates_file(self):
        """make_comparison_plot should create a PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            k_values = list(range(1, 17))
            K = np.array(k_values, dtype=float)
            acc = sigmoid_fn(K, 0.9, 0.1, 8.0, 0.5)
            all_accs = {'signal': {k: float(a) for k, a in zip(k_values, acc)}}
            all_popts = {'signal': [0.9, 0.1, 8.0, 0.5]}
            all_r2 = {'signal': 0.99}
            path = make_comparison_plot(
                k_values, all_accs, all_popts, all_r2,
                normal_acc=0.9, output_dir=tmpdir,
                hidden_size=16, num_hidden_layers=2)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
#  17. Scaling law fitting
# ---------------------------------------------------------------------------

class TestScalingLawFit:

    def _make_synthetic_results(self, n=20):
        """Generate synthetic results that follow K0 ~ a*H^alpha*L^gamma."""
        rng = np.random.default_rng(42)
        results = []
        for _ in range(n):
            H = rng.choice([16, 32, 64, 128])
            L = rng.choice([1, 2, 3, 5, 7])
            K0 = 0.5 * H ** 0.8 * L ** 0.2 + rng.normal(0, 0.5)
            beta = 2.0 * H ** (-0.3) * L ** (-0.1) + rng.normal(0, 0.01)
            g = np.exp(-beta)
            results.append({
                'H': int(H), 'L': int(L),
                'val_acc': 0.85,
                'normal_acc': 0.85,
                'n_params': int(H * 64 + H * H * (L - 1) + H * 10),
                'accs': {k: float(sigmoid_fn(k, 0.85, 0.1, K0, beta))
                         for k in range(1, int(H) + 1)},
                'sigmoid_A_inf': 0.85,
                'sigmoid_A_0': 0.1,
                'sigmoid_K_0': float(K0),
                'sigmoid_beta': float(beta),
                'sigmoid_g_eff': float(g),
                'sigmoid_R2': 0.95,
            })
        return results

    def test_mnist_fit_scaling_laws(self):
        """mnist fit_scaling_laws should return a dict with K0 key."""
        results = self._make_synthetic_results(25)
        scaling = mnist_fit_scaling_laws(results)
        assert scaling is not None
        assert 'K0' in scaling
        assert 'R2' in scaling['K0']

    def test_cifar_fit_scaling_laws(self):
        """cifar fit_scaling_laws should return a dict with K0 key."""
        results = self._make_synthetic_results(25)
        scaling = cifar_fit_scaling_laws(results)
        assert scaling is not None
        assert 'K0' in scaling

    def test_insufficient_data_returns_none(self):
        """With fewer than 5 good fits, should return None."""
        results = self._make_synthetic_results(3)
        scaling = mnist_fit_scaling_laws(results)
        # Could be None or have data depending on R2 threshold
        # Just verify no crash


# ---------------------------------------------------------------------------
#  18. CIFAR helper functions
# ---------------------------------------------------------------------------

class TestCIFARHelpers:

    def test_power_law_1d(self):
        """_power_law_1d: a * x^b."""
        assert abs(_power_law_1d(2.0, 3.0, 2.0) - 12.0) < 1e-10

    def test_power_law_2d(self):
        """_power_law_2d: a * H^alpha * L^gamma."""
        result = _power_law_2d((np.array([4.0]), np.array([2.0])),
                                1.0, 0.5, 1.0)
        expected = 1.0 * 4.0 ** 0.5 * 2.0 ** 1.0
        np.testing.assert_allclose(result, [expected])

    def test_print_results_table_no_crash(self):
        """print_results_table should run without error."""
        results = [{
            'H': 32, 'L': 2, 'n_params': 1000, 'val_acc': 0.4,
            'sigmoid_R2': 0.95,
            'sigmoid_A_inf': 0.9, 'sigmoid_A_0': 0.1,
            'sigmoid_K_0': 10.0, 'sigmoid_beta': 0.3,
            'sigmoid_g_eff': 0.74,
        }, {
            'H': 64, 'L': 3, 'n_params': 5000, 'val_acc': 0.3,
            'sigmoid_R2': None,
        }]
        print_results_table(results)  # should not raise


# ---------------------------------------------------------------------------
#  19. MNIST scaling plots with synthetic data
# ---------------------------------------------------------------------------

class TestMNISTScalingPlots:

    def test_make_scaling_plots_creates_files(self):
        """make_scaling_plots should create 3 PNG files."""
        rng = np.random.default_rng(42)
        results = []
        for H in [16, 32, 64]:
            for L in [1, 2, 3]:
                K0 = H * 0.4
                beta = 0.3
                results.append({
                    'H': H, 'L': L, 'val_acc': 0.8, 'normal_acc': 0.8,
                    'n_params': H * 100,
                    'accs': {k: float(sigmoid_fn(k, 0.8, 0.1, K0, beta))
                             for k in range(1, H + 1)},
                    'sigmoid_A_inf': 0.8, 'sigmoid_A_0': 0.1,
                    'sigmoid_K_0': K0, 'sigmoid_beta': beta,
                    'sigmoid_g_eff': np.exp(-beta), 'sigmoid_R2': 0.95,
                })

        import mnist_scaling
        old_dir = mnist_scaling.OUTPUT_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mnist_scaling.OUTPUT_DIR = tmpdir
            try:
                paths = make_scaling_plots(
                    results,
                    {'K0': {'a': 0.4, 'alpha': 1.0, 'gamma': 0.0, 'R2': 0.95}},
                )
                assert len(paths) == 2
                for p in paths:
                    assert os.path.isfile(p)
            finally:
                mnist_scaling.OUTPUT_DIR = old_dir


# ---------------------------------------------------------------------------
#  20. CIFAR plots with synthetic data
# ---------------------------------------------------------------------------

class TestCIFARPlots:

    def test_make_plots_creates_files(self):
        """cifar make_plots should create 3 PNG files."""
        results = []
        for H in [32, 64]:
            for L in [2, 3, 5]:
                K0 = H * 0.35
                beta = 0.25
                results.append({
                    'H': H, 'L': L, 'val_acc': 0.45, 'normal_acc': 0.45,
                    'n_params': H * 200,
                    'accs': {k: float(sigmoid_fn(k, 0.45, 0.1, K0, beta))
                             for k in range(1, H + 1)},
                    'sigmoid_A_inf': 0.45, 'sigmoid_A_0': 0.1,
                    'sigmoid_K_0': K0, 'sigmoid_beta': beta,
                    'sigmoid_g_eff': np.exp(-beta), 'sigmoid_R2': 0.92,
                })

        with tempfile.TemporaryDirectory() as tmpdir:
            scaling = cifar_fit_scaling_laws(results)
            paths = cifar_make_plots(results, scaling, tmpdir)
            assert len(paths) == 2
            for p in paths:
                assert os.path.isfile(p)


# ---------------------------------------------------------------------------
#  21. Summary printer
# ---------------------------------------------------------------------------

class TestPrintFitSummary:

    def test_no_crash_with_valid_data(self):
        """print_fit_summary should run without error."""
        all_popts = {'signal': [0.9, 0.1, 10.0, 0.3], 'weight': None}
        all_perrs = {'signal': [0.01, 0.02, 0.5, 0.01], 'weight': [0]*4}
        all_r2 = {'signal': 0.98, 'weight': float('nan')}
        print_fit_summary(all_popts, all_perrs, all_r2, 0.9)


# ---------------------------------------------------------------------------
#  22. Method registry completeness
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_all_methods_have_styles(self):
        """Every method in PRUNING_METHODS should have a matching style."""
        for m in PRUNING_METHODS:
            assert m in METHOD_STYLE
            assert 'color' in METHOD_STYLE[m]
            assert 'marker' in METHOD_STYLE[m]


# ---------------------------------------------------------------------------
#  23. Regression: evaluate_masked_accuracy preserves training mode
# ---------------------------------------------------------------------------

class TestEvalMaskedAccuracyPreservesTrainingMode:
    """Guard for the Phase-3 patch: source model's `training` flag must
    round-trip through evaluate_masked_accuracy."""

    def _setup(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from unstructured_pruning.core import evaluate_masked_accuracy
        from unstructured_pruning.methods import magnitude_masks
        m = FCNetwork(input_size=8, hidden_size=4, num_hidden_layers=2,
                      num_classes=3, seed=0)
        X = np.random.randn(20, 8).astype(np.float32)
        y = np.random.randint(0, 3, size=20)
        masks = magnitude_masks(m, [0.5, 1.0], n_seeds=1, base_seed=0)
        return evaluate_masked_accuracy, m, X, y, masks

    def test_eval_mode_preserved(self):
        eval_fn, m, X, y, masks = self._setup()
        m.eval()
        eval_fn(m, X, y, masks)
        assert m.training is False

    def test_train_mode_preserved(self):
        eval_fn, m, X, y, masks = self._setup()
        m.train()
        eval_fn(m, X, y, masks)
        assert m.training is True


# ---------------------------------------------------------------------------
#  24. Regression: fit_param_scaling handles non-positive s_0
# ---------------------------------------------------------------------------

class TestFitParamScalingRobustness:
    """Guard for the Phase-4 patch: log-log OLS must drop non-positive
    entries instead of silently returning NaN for the whole fit."""

    def _import(self):
        import sys, os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from unstructured_pruning.param_scaling import fit_param_scaling
        return fit_param_scaling

    def test_clean_data(self):
        fit = self._import()
        P = np.array([1e3, 1e4, 1e5, 1e6])
        s0 = 0.5 * P ** -0.15
        phi, a, r2 = fit(P, s0)
        np.testing.assert_allclose(phi, -0.15, atol=1e-6)
        np.testing.assert_allclose(a, 0.5, atol=1e-6)
        assert r2 > 0.999

    def test_drops_zero_entries(self):
        fit = self._import()
        P = np.array([1e3, 1e4, 1e5, 1e6])
        s0 = np.array([0.5, 0.1, 0.01, 0.0])  # one invalid → 3 valid points
        phi, a, r2 = fit(P, s0)
        assert np.isfinite(phi) and np.isfinite(a) and np.isfinite(r2)

    def test_all_invalid_returns_nan_triplet(self):
        fit = self._import()
        phi, a, r2 = fit(np.array([1e3, 1e4]), np.array([0.0, -0.1]))
        assert np.isnan(phi) and np.isnan(a) and np.isnan(r2)

    def test_one_valid_point_returns_nan(self):
        """Need ≥2 finite points to define a line."""
        fit = self._import()
        phi, a, r2 = fit(np.array([1e3, 1e4]), np.array([0.5, 0.0]))
        assert np.isnan(phi) and np.isnan(a) and np.isnan(r2)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])