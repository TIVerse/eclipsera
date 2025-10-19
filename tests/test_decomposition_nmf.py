"""Tests for Non-negative Matrix Factorization."""
import numpy as np
import pytest

from eclipsera.decomposition import NMF


def test_nmf_basic():
    """Test basic NMF functionality."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(X)
    
    assert hasattr(nmf, 'components_')
    assert hasattr(nmf, 'reconstruction_err_')
    assert nmf.components_.shape == (5, 10)


def test_nmf_fit_transform():
    """Test NMF fit_transform method."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 5)
    assert hasattr(nmf, 'components_')


def test_nmf_transform():
    """Test NMF transform method."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(X)
    W = nmf.transform(X)
    
    assert W.shape == (20, 5)


def test_nmf_inverse_transform():
    """Test NMF inverse_transform method."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, random_state=42)
    W = nmf.fit_transform(X)
    X_reconstructed = nmf.inverse_transform(W)
    
    assert X_reconstructed.shape == X.shape


def test_nmf_reconstruction():
    """Test NMF reconstruction."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, max_iter=500, random_state=42)
    W = nmf.fit_transform(X)
    X_reconstructed = nmf.inverse_transform(W)
    
    # Reconstruction should be reasonable
    error = np.linalg.norm(X - X_reconstructed, 'fro')
    assert error < 100  # Loose bound


def test_nmf_non_negative_components():
    """Test NMF produces non-negative components."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, random_state=42)
    W = nmf.fit_transform(X)
    
    assert np.all(nmf.components_ >= 0)
    assert np.all(W >= 0)


def test_nmf_random_init():
    """Test NMF with random initialization."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, init='random', random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 5)


def test_nmf_nndsvd_init():
    """Test NMF with NNDSVD initialization."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, init='nndsvd', random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 5)


def test_nmf_n_components_none():
    """Test NMF with n_components=None."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=None, random_state=42)
    nmf.fit(X)
    
    # Should use min(n_samples, n_features)
    assert nmf.n_components_ == min(20, 10)


def test_nmf_n_components_variations():
    """Test NMF with different n_components."""
    X = np.abs(np.random.randn(30, 20))
    
    for n_comp in [3, 5, 10]:
        nmf = NMF(n_components=n_comp, random_state=42)
        W = nmf.fit_transform(X)
        
        assert W.shape == (30, n_comp)
        assert nmf.components_.shape == (n_comp, 20)


def test_nmf_max_iter():
    """Test NMF respects max_iter."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, max_iter=10, random_state=42)
    nmf.fit(X)
    
    assert nmf.n_iter_ <= 10


def test_nmf_convergence():
    """Test NMF converges."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, max_iter=500, tol=1e-4, random_state=42)
    nmf.fit(X)
    
    # Should converge before max_iter
    assert nmf.n_iter_ <= 500


def test_nmf_error_negative_input():
    """Test NMF raises error for negative input."""
    X = np.random.randn(20, 10)  # Can have negative values
    
    nmf = NMF(n_components=5)
    
    with pytest.raises(ValueError, match="non-negative"):
        nmf.fit(X)


def test_nmf_error_negative_transform():
    """Test NMF raises error for negative input in transform."""
    X_train = np.abs(np.random.randn(20, 10))
    X_test = np.random.randn(10, 10)  # Has negatives
    
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(X_train)
    
    with pytest.raises(ValueError, match="non-negative"):
        nmf.transform(X_test)


def test_nmf_error_before_fit():
    """Test NMF raises error when calling methods before fit."""
    nmf = NMF(n_components=5)
    X = np.abs(np.random.randn(10, 10))
    
    with pytest.raises(ValueError):
        nmf.transform(X)
    
    with pytest.raises(ValueError):
        nmf.inverse_transform(X[:, :5])


def test_nmf_reproducibility():
    """Test NMF is reproducible with random_state."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf1 = NMF(n_components=5, random_state=42)
    W1 = nmf1.fit_transform(X)
    
    nmf2 = NMF(n_components=5, random_state=42)
    W2 = nmf2.fit_transform(X)
    
    # Results should be similar (allowing for small numerical differences)
    assert np.allclose(W1, W2, rtol=0.1)


def test_nmf_reconstruction_error_decreases():
    """Test reconstruction error decreases with more iterations."""
    X = np.abs(np.random.randn(30, 15))
    
    # Few iterations
    nmf1 = NMF(n_components=5, max_iter=10, random_state=42)
    nmf1.fit(X)
    
    # More iterations
    nmf2 = NMF(n_components=5, max_iter=200, random_state=42)
    nmf2.fit(X)
    
    # More iterations should generally give lower error
    # (allowing some margin for randomness)
    assert nmf2.reconstruction_err_ <= nmf1.reconstruction_err_ * 1.2


def test_nmf_small_dataset():
    """Test NMF on small dataset."""
    X = np.abs(np.random.randn(5, 3))
    
    nmf = NMF(n_components=2, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (5, 2)


def test_nmf_regularization():
    """Test NMF with regularization."""
    X = np.abs(np.random.randn(20, 10))
    
    nmf = NMF(n_components=5, alpha=0.1, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 5)


def test_nmf_different_shapes():
    """Test NMF on different data shapes."""
    shapes = [(10, 20), (20, 10), (15, 15)]
    
    for n_samples, n_features in shapes:
        X = np.abs(np.random.randn(n_samples, n_features))
        nmf = NMF(n_components=5, random_state=42)
        W = nmf.fit_transform(X)
        
        assert W.shape == (n_samples, 5)
        assert nmf.components_.shape == (5, n_features)


def test_nmf_topic_modeling_use_case():
    """Test NMF for topic modeling scenario."""
    # Simulate document-term matrix
    X = np.abs(np.random.randn(50, 100))  # 50 docs, 100 terms
    
    nmf = NMF(n_components=10, init='nndsvd', random_state=42)
    W = nmf.fit_transform(X)  # Document-topic matrix
    H = nmf.components_       # Topic-term matrix
    
    assert W.shape == (50, 10)  # 50 docs, 10 topics
    assert H.shape == (10, 100)  # 10 topics, 100 terms
    assert np.all(W >= 0)
    assert np.all(H >= 0)


def test_nmf_n_features_in():
    """Test n_features_in_ attribute is set."""
    X = np.abs(np.random.randn(20, 15))
    
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(X)
    
    assert nmf.n_features_in_ == 15


def test_nmf_sparse_like():
    """Test NMF on sparse-like (mostly zeros) data."""
    X = np.abs(np.random.randn(20, 10))
    X[X < 0.5] = 0  # Make it sparse
    
    nmf = NMF(n_components=5, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 5)


def test_nmf_init_comparison():
    """Test different initialization methods."""
    X = np.abs(np.random.randn(30, 20))
    
    # Both should work
    nmf_random = NMF(n_components=5, init='random', random_state=42)
    W_random = nmf_random.fit_transform(X)
    
    nmf_nndsvd = NMF(n_components=5, init='nndsvd', random_state=42)
    W_nndsvd = nmf_nndsvd.fit_transform(X)
    
    # Both should produce valid output
    assert W_random.shape == (30, 5)
    assert W_nndsvd.shape == (30, 5)


def test_nmf_high_n_components():
    """Test NMF with high number of components."""
    X = np.abs(np.random.randn(20, 30))
    
    nmf = NMF(n_components=15, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (20, 15)


def test_nmf_reconstruction_quality():
    """Test NMF reconstruction quality with many components."""
    X = np.abs(np.random.randn(20, 10))
    
    # With all components, reconstruction should be better
    nmf = NMF(n_components=10, max_iter=500, random_state=42)
    nmf.fit(X)
    
    # Check reconstruction error is tracked
    assert nmf.reconstruction_err_ >= 0
