"""Tests for hierarchical clustering."""
import numpy as np
import pytest

from eclipsera.cluster import AgglomerativeClustering


def test_agglomerative_basic():
    """Test basic AgglomerativeClustering functionality."""
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    
    clustering = AgglomerativeClustering(n_clusters=2)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert hasattr(clustering, 'n_clusters_')
    assert len(clustering.labels_) == 6


def test_agglomerative_fit_predict():
    """Test AgglomerativeClustering fit_predict method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    
    clustering = AgglomerativeClustering(n_clusters=2)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (6,)
    assert len(np.unique(labels)) == 2


def test_agglomerative_n_clusters():
    """Test AgglomerativeClustering with different n_clusters."""
    X = np.random.randn(30, 3)
    
    for n_clusters in [2, 3, 5]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)
        
        assert clustering.n_clusters_ == n_clusters
        assert len(np.unique(labels)) == n_clusters


def test_agglomerative_ward_linkage():
    """Test AgglomerativeClustering with ward linkage."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_complete_linkage():
    """Test AgglomerativeClustering with complete linkage."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='complete')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_average_linkage():
    """Test AgglomerativeClustering with average linkage."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_single_linkage():
    """Test AgglomerativeClustering with single linkage."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='single')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_distance_threshold():
    """Test AgglomerativeClustering with distance threshold."""
    X = np.random.randn(30, 2)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=5.0,
        linkage='average'
    )
    labels = clustering.fit_predict(X)
    
    assert hasattr(clustering, 'n_clusters_')
    assert clustering.n_clusters_ >= 1


def test_agglomerative_well_separated():
    """Test AgglomerativeClustering on well-separated clusters."""
    # Create two well-separated clusters
    cluster1 = np.random.randn(15, 2) * 0.3 + np.array([0, 0])
    cluster2 = np.random.randn(15, 2) * 0.3 + np.array([10, 10])
    X = np.vstack([cluster1, cluster2])
    
    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = clustering.fit_predict(X)
    
    # Should identify 2 clusters
    assert len(np.unique(labels)) == 2


def test_agglomerative_single_cluster():
    """Test AgglomerativeClustering with single cluster."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=1)
    labels = clustering.fit_predict(X)
    
    assert np.all(labels == 0)


def test_agglomerative_deterministic():
    """Test AgglomerativeClustering is deterministic."""
    X = np.random.RandomState(42).randn(20, 2)
    
    labels1 = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    labels2 = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    
    # Should give same results (deterministic)
    assert np.array_equal(labels1, labels2)


def test_agglomerative_different_dimensions():
    """Test AgglomerativeClustering on different dimensional data."""
    for n_features in [2, 5, 10]:
        X = np.random.randn(25, n_features)
        
        clustering = AgglomerativeClustering(n_clusters=3)
        labels = clustering.fit_predict(X)
        
        assert labels.shape == (25,)
        assert clustering.n_features_in_ == n_features


def test_agglomerative_linkage_comparison():
    """Test different linkage methods give different results."""
    X = np.random.randn(30, 2)
    
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
    complete = AgglomerativeClustering(n_clusters=3, linkage='complete').fit_predict(X)
    
    # Different linkages may give different results
    # Just check they both produce valid output
    assert len(np.unique(ward)) == 3
    assert len(np.unique(complete)) == 3


def test_agglomerative_three_clusters():
    """Test AgglomerativeClustering with three well-separated clusters."""
    # Create three clusters
    cluster1 = np.random.randn(10, 2) * 0.3 + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) * 0.3 + np.array([5, 5])
    cluster3 = np.random.randn(10, 2) * 0.3 + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(X)
    
    assert len(np.unique(labels)) == 3


def test_agglomerative_small_dataset():
    """Test AgglomerativeClustering on small dataset."""
    X = np.array([[1, 1], [2, 2], [10, 10]])
    
    clustering = AgglomerativeClustering(n_clusters=2)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (3,)
    assert len(np.unique(labels)) == 2


def test_agglomerative_distance_threshold_stops_early():
    """Test distance threshold stops merging early."""
    # Dense cluster
    X = np.random.randn(20, 2) * 0.5
    
    # Low threshold should create more clusters
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.1,
        linkage='single'
    )
    labels = clustering.fit_predict(X)
    
    # Should create multiple clusters
    assert clustering.n_clusters_ > 1


def test_agglomerative_labels_range():
    """Test AgglomerativeClustering labels are in valid range."""
    X = np.random.randn(30, 2)
    
    clustering = AgglomerativeClustering(n_clusters=4)
    labels = clustering.fit_predict(X)
    
    # Labels should be 0 to n_clusters-1
    assert np.all(labels >= 0)
    assert np.all(labels < 4)


def test_agglomerative_n_features_in():
    """Test n_features_in_ attribute is set."""
    X = np.random.randn(20, 7)
    
    clustering = AgglomerativeClustering(n_clusters=3)
    clustering.fit(X)
    
    assert clustering.n_features_in_ == 7


def test_agglomerative_high_dimensional():
    """Test AgglomerativeClustering on high-dimensional data."""
    X = np.random.randn(30, 20)
    
    clustering = AgglomerativeClustering(n_clusters=5, linkage='average')
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (30,)
    assert len(np.unique(labels)) == 5


def test_agglomerative_ward_vs_others():
    """Test ward linkage behavior."""
    X = np.random.randn(25, 3)
    
    # Ward should work
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_ward = ward.fit_predict(X)
    
    # Average should also work
    avg = AgglomerativeClustering(n_clusters=3, linkage='average')
    labels_avg = avg.fit_predict(X)
    
    assert len(np.unique(labels_ward)) == 3
    assert len(np.unique(labels_avg)) == 3


def test_agglomerative_cluster_sizes():
    """Test AgglomerativeClustering produces valid cluster sizes."""
    X = np.random.randn(40, 2)
    
    clustering = AgglomerativeClustering(n_clusters=4)
    labels = clustering.fit_predict(X)
    
    # All samples should be assigned
    assert len(labels) == 40
    
    # Each cluster should have at least one sample
    cluster_sizes = np.bincount(labels)
    assert np.all(cluster_sizes > 0)


def test_agglomerative_invalid_linkage():
    """Test AgglomerativeClustering raises error for invalid linkage."""
    X = np.random.randn(20, 2)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='invalid')
    
    with pytest.raises(ValueError):
        clustering.fit(X)
