"""Tests for t-SNE manifold learning."""
import numpy as np
import pytest

from eclipsera.manifold import TSNE


def test_tsne_basic():
    """Test basic TSNE functionality."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    assert X_embedded.shape == (50, 2)
    assert hasattr(tsne, 'embedding_')
    assert hasattr(tsne, 'kl_divergence_')


def test_tsne_fit():
    """Test TSNE fit method."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    tsne.fit(X)
    
    assert hasattr(tsne, 'embedding_')
    assert tsne.embedding_.shape == (50, 2)


def test_tsne_fit_transform():
    """Test TSNE fit_transform method."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    assert X_embedded.shape == (50, 2)
    assert np.array_equal(X_embedded, tsne.embedding_)


def test_tsne_n_components():
    """Test TSNE with different n_components."""
    X = np.random.randn(50, 10)
    
    for n_comp in [2, 3]:
        tsne = TSNE(n_components=n_comp, n_iter=250, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        assert X_embedded.shape == (50, n_comp)


def test_tsne_perplexity():
    """Test TSNE with different perplexity values."""
    X = np.random.randn(50, 10)
    
    for perplexity in [5, 15, 30]:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=250, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        assert X_embedded.shape == (50, 2)


def test_tsne_learning_rate():
    """Test TSNE with different learning rates."""
    X = np.random.randn(50, 10)
    
    for lr in [100.0, 200.0, 500.0]:
        tsne = TSNE(n_components=2, learning_rate=lr, n_iter=250, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        assert X_embedded.shape == (50, 2)


def test_tsne_n_iter():
    """Test TSNE respects n_iter."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=100, random_state=42)
    tsne.fit(X)
    
    assert tsne.n_iter_ == 100


def test_tsne_kl_divergence():
    """Test TSNE computes KL divergence."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    tsne.fit(X)
    
    assert hasattr(tsne, 'kl_divergence_')
    assert tsne.kl_divergence_ >= 0


def test_tsne_reproducibility():
    """Test TSNE is reproducible with random_state."""
    X = np.random.RandomState(42).randn(40, 8)
    
    tsne1 = TSNE(n_components=2, n_iter=250, random_state=42)
    X1 = tsne1.fit_transform(X)
    
    tsne2 = TSNE(n_components=2, n_iter=250, random_state=42)
    X2 = tsne2.fit_transform(X)
    
    # Results should be very similar
    assert np.allclose(X1, X2, rtol=0.1)


def test_tsne_small_dataset():
    """Test TSNE on small dataset."""
    X = np.random.randn(20, 5)
    
    tsne = TSNE(n_components=2, perplexity=5, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    assert X_embedded.shape == (20, 2)


def test_tsne_error_perplexity_too_large():
    """Test TSNE raises error when perplexity >= n_samples."""
    X = np.random.randn(20, 5)
    
    tsne = TSNE(n_components=2, perplexity=25, n_iter=250)
    
    with pytest.raises(ValueError, match="perplexity must be less than n_samples"):
        tsne.fit(X)


def test_tsne_n_features_in():
    """Test n_features_in_ attribute is set."""
    X = np.random.randn(50, 12)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    tsne.fit(X)
    
    assert tsne.n_features_in_ == 12


def test_tsne_embedding_centered():
    """Test TSNE embedding is centered."""
    X = np.random.randn(50, 10)
    
    tsne = TSNE(n_components=2, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Embedding should be roughly centered
    assert np.allclose(X_embedded.mean(axis=0), 0, atol=0.1)


def test_tsne_different_dimensions():
    """Test TSNE on different dimensional data."""
    for n_features in [5, 10, 20]:
        X = np.random.randn(40, n_features)
        
        tsne = TSNE(n_components=2, n_iter=250, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        assert X_embedded.shape == (40, 2)
        assert tsne.n_features_in_ == n_features


def test_tsne_verbose_mode():
    """Test TSNE with verbose mode."""
    X = np.random.randn(40, 8)
    
    tsne = TSNE(n_components=2, n_iter=300, verbose=1, random_state=42)
    
    # Should not raise error with verbose
    X_embedded = tsne.fit_transform(X)
    assert X_embedded.shape == (40, 2)


def test_tsne_preserves_local_structure():
    """Test TSNE preserves some local structure."""
    # Create data with clear structure
    cluster1 = np.random.randn(20, 10) * 0.5
    cluster2 = np.random.randn(20, 10) * 0.5 + 5
    X = np.vstack([cluster1, cluster2])
    
    tsne = TSNE(n_components=2, perplexity=15, n_iter=500, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Check that clusters are somewhat separated in embedding
    dist_within1 = np.mean(np.linalg.norm(X_embedded[:20] - X_embedded[:20].mean(axis=0), axis=1))
    dist_within2 = np.mean(np.linalg.norm(X_embedded[20:] - X_embedded[20:].mean(axis=0), axis=1))
    dist_between = np.linalg.norm(X_embedded[:20].mean(axis=0) - X_embedded[20:].mean(axis=0))
    
    # Between-cluster distance should be larger than within-cluster
    assert dist_between > dist_within1
    assert dist_between > dist_within2


def test_tsne_3d_embedding():
    """Test TSNE with 3D embedding."""
    X = np.random.randn(50, 15)
    
    tsne = TSNE(n_components=3, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    assert X_embedded.shape == (50, 3)


def test_tsne_kl_decreases_with_iterations():
    """Test that more iterations generally give better KL divergence."""
    X = np.random.randn(40, 8)
    
    # Few iterations
    tsne1 = TSNE(n_components=2, n_iter=100, random_state=42)
    tsne1.fit(X)
    
    # More iterations
    tsne2 = TSNE(n_components=2, n_iter=500, random_state=42)
    tsne2.fit(X)
    
    # More iterations should generally give lower or similar KL
    # (allowing some margin for randomness)
    assert tsne2.kl_divergence_ <= tsne1.kl_divergence_ * 1.5


def test_tsne_high_dimensional():
    """Test TSNE on high-dimensional data."""
    X = np.random.randn(50, 50)
    
    tsne = TSNE(n_components=2, perplexity=20, n_iter=250, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    assert X_embedded.shape == (50, 2)
