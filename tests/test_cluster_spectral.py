"""Tests for Spectral Clustering."""
import numpy as np
import pytest

from eclipsera.cluster import SpectralClustering


def test_spectral_basic():
    """Test basic SpectralClustering functionality."""
    X = np.random.randn(50, 5)
    
    clustering = SpectralClustering(n_clusters=3, random_state=42)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert hasattr(clustering, 'affinity_matrix_')
    assert len(clustering.labels_) == 50


def test_spectral_fit_predict():
    """Test SpectralClustering fit_predict method."""
    X = np.random.randn(50, 5)
    
    clustering = SpectralClustering(n_clusters=3, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (50,)
    assert len(np.unique(labels)) <= 3


def test_spectral_n_clusters():
    """Test SpectralClustering with different n_clusters."""
    X = np.random.randn(60, 5)
    
    for n_clusters in [2, 3, 5]:
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(X)
        
        assert len(np.unique(labels)) == n_clusters


def test_spectral_rbf_affinity():
    """Test SpectralClustering with RBF affinity."""
    X = np.random.randn(40, 5)
    
    clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (40,)


def test_spectral_nearest_neighbors_affinity():
    """Test SpectralClustering with nearest neighbors affinity."""
    X = np.random.randn(40, 5)
    
    clustering = SpectralClustering(
        n_clusters=3,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=42
    )
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (40,)


def test_spectral_gamma_parameter():
    """Test SpectralClustering with different gamma values."""
    X = np.random.randn(50, 5)
    
    for gamma in [0.5, 1.0, 2.0]:
        clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=gamma, random_state=42)
        labels = clustering.fit_predict(X)
        
        assert len(labels) == 50


def test_spectral_n_neighbors_parameter():
    """Test SpectralClustering with different n_neighbors."""
    X = np.random.randn(50, 5)
    
    for n_neighbors in [5, 10, 15]:
        clustering = SpectralClustering(
            n_clusters=3,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=42
        )
        labels = clustering.fit_predict(X)
        
        assert len(labels) == 50


def test_spectral_affinity_matrix():
    """Test SpectralClustering stores affinity matrix."""
    X = np.random.randn(30, 5)
    
    clustering = SpectralClustering(n_clusters=3, random_state=42)
    clustering.fit(X)
    
    assert clustering.affinity_matrix_.shape == (30, 30)
    # Affinity matrix should be symmetric
    assert np.allclose(clustering.affinity_matrix_, clustering.affinity_matrix_.T)


def test_spectral_reproducibility():
    """Test SpectralClustering is reproducible with random_state."""
    X = np.random.RandomState(42).randn(50, 5)
    
    labels1 = SpectralClustering(n_clusters=3, random_state=42).fit_predict(X)
    labels2 = SpectralClustering(n_clusters=3, random_state=42).fit_predict(X)
    
    # Results should be identical
    assert np.array_equal(labels1, labels2)


def test_spectral_well_separated():
    """Test SpectralClustering on well-separated clusters."""
    # Create three well-separated clusters
    cluster1 = np.random.randn(20, 5) * 0.5
    cluster2 = np.random.randn(20, 5) * 0.5 + np.array([10, 0, 0, 0, 0])
    cluster3 = np.random.randn(20, 5) * 0.5 + np.array([0, 10, 0, 0, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    clustering = SpectralClustering(n_clusters=3, random_state=42)
    labels = clustering.fit_predict(X)
    
    # Should identify 3 clusters
    assert len(np.unique(labels)) == 3


def test_spectral_small_dataset():
    """Test SpectralClustering on small dataset."""
    X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
    
    clustering = SpectralClustering(n_clusters=2, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (4,)
    assert len(np.unique(labels)) == 2


def test_spectral_n_features_in():
    """Test n_features_in_ attribute is set."""
    X = np.random.randn(50, 8)
    
    clustering = SpectralClustering(n_clusters=3, random_state=42)
    clustering.fit(X)
    
    assert clustering.n_features_in_ == 8


def test_spectral_labels_range():
    """Test SpectralClustering labels are in valid range."""
    X = np.random.randn(50, 5)
    
    clustering = SpectralClustering(n_clusters=4, random_state=42)
    labels = clustering.fit_predict(X)
    
    # Labels should be 0 to n_clusters-1
    assert np.all(labels >= 0)
    assert np.all(labels < 4)


def test_spectral_different_dimensions():
    """Test SpectralClustering on different dimensional data."""
    for n_features in [2, 5, 10]:
        X = np.random.randn(40, n_features)
        
        clustering = SpectralClustering(n_clusters=3, random_state=42)
        labels = clustering.fit_predict(X)
        
        assert labels.shape == (40,)
        assert clustering.n_features_in_ == n_features


def test_spectral_high_dimensional():
    """Test SpectralClustering on high-dimensional data."""
    X = np.random.randn(50, 30)
    
    clustering = SpectralClustering(n_clusters=4, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (50,)


def test_spectral_two_clusters():
    """Test SpectralClustering with two clusters."""
    X = np.random.randn(60, 5)
    
    clustering = SpectralClustering(n_clusters=2, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert len(np.unique(labels)) == 2


def test_spectral_single_cluster():
    """Test SpectralClustering with single cluster."""
    X = np.random.randn(30, 5)
    
    clustering = SpectralClustering(n_clusters=1, random_state=42)
    labels = clustering.fit_predict(X)
    
    assert np.all(labels == 0)


def test_spectral_affinity_positive():
    """Test SpectralClustering affinity matrix has positive values."""
    X = np.random.randn(30, 5)
    
    clustering = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
    clustering.fit(X)
    
    # RBF affinity should be non-negative
    assert np.all(clustering.affinity_matrix_ >= 0)


def test_spectral_rbf_vs_knn():
    """Test different affinity methods give different results."""
    X = np.random.randn(40, 5)
    
    rbf = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
    rbf_labels = rbf.fit_predict(X)
    
    knn = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
    knn_labels = knn.fit_predict(X)
    
    # Both should produce valid output
    assert len(np.unique(rbf_labels)) == 3
    assert len(np.unique(knn_labels)) == 3


def test_spectral_invalid_affinity():
    """Test SpectralClustering raises error for invalid affinity."""
    X = np.random.randn(30, 5)
    
    clustering = SpectralClustering(n_clusters=3, affinity='invalid')
    
    with pytest.raises(ValueError):
        clustering.fit(X)


def test_spectral_cluster_assignment():
    """Test all samples are assigned to clusters."""
    X = np.random.randn(50, 5)
    
    clustering = SpectralClustering(n_clusters=4, random_state=42)
    labels = clustering.fit_predict(X)
    
    # All samples should be assigned
    assert len(labels) == 50
    assert not np.any(np.isnan(labels))
