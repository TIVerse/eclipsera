"""Tests for DBSCAN clustering."""
import numpy as np
import pytest

from eclipsera.cluster import DBSCAN


def test_dbscan_basic():
    """Test basic DBSCAN functionality."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert hasattr(clustering, 'core_sample_indices_')
    assert hasattr(clustering, 'components_')
    assert len(clustering.labels_) == 6


def test_dbscan_fit_predict():
    """Test DBSCAN fit_predict method."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    
    clustering = DBSCAN(eps=3, min_samples=2)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (6,)
    # Should have at least one noise point (-1)
    assert -1 in labels


def test_dbscan_identifies_clusters():
    """Test DBSCAN identifies clusters correctly."""
    # Create two clear clusters
    cluster1 = np.random.randn(20, 2) * 0.3
    cluster2 = np.random.randn(20, 2) * 0.3 + np.array([10, 10])
    X = np.vstack([cluster1, cluster2])
    
    clustering = DBSCAN(eps=1.0, min_samples=5)
    labels = clustering.fit_predict(X)
    
    # Should identify at least 2 clusters
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    assert len(unique_labels) >= 2


def test_dbscan_noise_detection():
    """Test DBSCAN detects noise points."""
    # Create cluster with outliers
    cluster = np.random.randn(20, 2) * 0.5
    outliers = np.array([[10, 10], [15, 15], [-10, -10]])
    X = np.vstack([cluster, outliers])
    
    clustering = DBSCAN(eps=1.5, min_samples=3)
    labels = clustering.fit_predict(X)
    
    # Should have noise points
    assert np.any(labels == -1)


def test_dbscan_eps_parameter():
    """Test DBSCAN with different eps values."""
    X = np.random.randn(50, 2)
    
    # Small eps - more clusters/noise
    dbscan1 = DBSCAN(eps=0.3, min_samples=5)
    labels1 = dbscan1.fit_predict(X)
    
    # Large eps - fewer clusters
    dbscan2 = DBSCAN(eps=2.0, min_samples=5)
    labels2 = dbscan2.fit_predict(X)
    
    # Larger eps should generally mean fewer noise points
    noise_count1 = np.sum(labels1 == -1)
    noise_count2 = np.sum(labels2 == -1)
    
    assert noise_count2 <= noise_count1


def test_dbscan_min_samples_parameter():
    """Test DBSCAN with different min_samples values."""
    X = np.random.randn(50, 2)
    
    for min_samples in [2, 5, 10]:
        clustering = DBSCAN(eps=1.0, min_samples=min_samples)
        labels = clustering.fit_predict(X)
        
        assert labels.shape == (50,)


def test_dbscan_core_samples():
    """Test DBSCAN core samples are stored correctly."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(X)
    
    assert len(clustering.core_sample_indices_) > 0
    assert clustering.components_.shape[0] == len(clustering.core_sample_indices_)


def test_dbscan_euclidean_metric():
    """Test DBSCAN with Euclidean metric."""
    X = np.random.randn(30, 2)
    
    clustering = DBSCAN(eps=1.0, min_samples=5, metric='euclidean')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (30,)


def test_dbscan_manhattan_metric():
    """Test DBSCAN with Manhattan metric."""
    X = np.random.randn(30, 2)
    
    clustering = DBSCAN(eps=1.5, min_samples=5, metric='manhattan')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (30,)


def test_dbscan_single_cluster():
    """Test DBSCAN with dense single cluster."""
    # Create a single dense cluster
    X = np.random.randn(50, 2) * 0.5
    
    clustering = DBSCAN(eps=2.0, min_samples=3)
    labels = clustering.fit_predict(X)
    
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    # Should identify one main cluster
    assert len(unique_labels) >= 1


def test_dbscan_no_clusters():
    """Test DBSCAN when no clusters form."""
    # Sparse random points
    X = np.random.randn(10, 2) * 10
    
    clustering = DBSCAN(eps=0.5, min_samples=5)
    labels = clustering.fit_predict(X)
    
    # Most/all should be noise
    assert np.sum(labels == -1) > 0


def test_dbscan_different_dimensions():
    """Test DBSCAN on different dimensional data."""
    for n_features in [2, 5, 10]:
        X = np.random.randn(40, n_features)
        
        clustering = DBSCAN(eps=2.0, min_samples=3)
        labels = clustering.fit_predict(X)
        
        assert labels.shape == (40,)
        assert clustering.n_features_in_ == n_features


def test_dbscan_high_density():
    """Test DBSCAN on high-density data."""
    # Very dense cluster
    X = np.random.randn(100, 2) * 0.1
    
    clustering = DBSCAN(eps=0.5, min_samples=5)
    labels = clustering.fit_predict(X)
    
    # Most points should be in a cluster (not noise)
    assert np.sum(labels != -1) > 80


def test_dbscan_well_separated():
    """Test DBSCAN on well-separated clusters."""
    # Three well-separated clusters
    cluster1 = np.random.randn(15, 2) * 0.3 + np.array([0, 0])
    cluster2 = np.random.randn(15, 2) * 0.3 + np.array([10, 0])
    cluster3 = np.random.randn(15, 2) * 0.3 + np.array([5, 10])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    clustering = DBSCAN(eps=1.0, min_samples=3)
    labels = clustering.fit_predict(X)
    
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    # Should identify 3 clusters
    assert len(unique_labels) >= 3


def test_dbscan_components_shape():
    """Test DBSCAN components have correct shape."""
    X = np.random.randn(50, 3)
    
    clustering = DBSCAN(eps=1.0, min_samples=5)
    clustering.fit(X)
    
    if len(clustering.core_sample_indices_) > 0:
        assert clustering.components_.shape[1] == 3


def test_dbscan_labels_range():
    """Test DBSCAN labels are in valid range."""
    X = np.random.randn(50, 2)
    
    clustering = DBSCAN(eps=1.0, min_samples=5)
    labels = clustering.fit_predict(X)
    
    # Labels should be >= -1 (noise) and < n_samples
    assert np.all(labels >= -1)
    assert np.all(labels < len(X))


def test_dbscan_empty_components():
    """Test DBSCAN when no core samples found."""
    # Very strict parameters
    X = np.random.randn(10, 2) * 5
    
    clustering = DBSCAN(eps=0.1, min_samples=10)
    clustering.fit(X)
    
    # May have no core samples
    assert len(clustering.core_sample_indices_) == 0
    assert clustering.components_.shape[0] == 0


def test_dbscan_reproducibility():
    """Test DBSCAN is deterministic."""
    X = np.random.RandomState(42).randn(50, 2)
    
    labels1 = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    labels2 = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    
    # Should give same results (deterministic algorithm)
    assert np.array_equal(labels1, labels2)


def test_dbscan_vs_kmeans_use_case():
    """Test DBSCAN handles non-spherical clusters better than K-means."""
    # Create a crescent-shaped cluster (non-spherical)
    t = np.linspace(0, np.pi, 50)
    X = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(50, 2) * 0.1
    
    clustering = DBSCAN(eps=0.5, min_samples=3)
    labels = clustering.fit_predict(X)
    
    # Should identify as a single cluster (not noise)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    assert len(unique_labels) >= 1
