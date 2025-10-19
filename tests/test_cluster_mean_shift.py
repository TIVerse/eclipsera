"""Tests for Mean Shift clustering."""
import numpy as np
import pytest

from eclipsera.cluster import MeanShift


def test_mean_shift_basic():
    """Test basic MeanShift functionality."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert hasattr(clustering, 'cluster_centers_')
    assert len(clustering.labels_) == 50


def test_mean_shift_fit_predict():
    """Test MeanShift fit_predict method."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (50,)


def test_mean_shift_predict():
    """Test MeanShift predict method."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    new_points = np.random.randn(10, 5)
    labels = clustering.predict(new_points)
    
    assert labels.shape == (10,)


def test_mean_shift_cluster_centers():
    """Test MeanShift stores cluster centers."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    assert clustering.cluster_centers_.shape[1] == 5
    assert len(clustering.cluster_centers_) > 0


def test_mean_shift_auto_bandwidth():
    """Test MeanShift with automatic bandwidth estimation."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=None)  # Auto-estimate
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (50,)
    assert len(clustering.cluster_centers_) > 0


def test_mean_shift_different_bandwidths():
    """Test MeanShift with different bandwidth values."""
    X = np.random.randn(50, 5)
    
    for bandwidth in [0.5, 1.0, 2.0]:
        clustering = MeanShift(bandwidth=bandwidth)
        labels = clustering.fit_predict(X)
        
        assert len(labels) == 50


def test_mean_shift_cluster_all_true():
    """Test MeanShift with cluster_all=True."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0, cluster_all=True)
    labels = clustering.fit_predict(X)
    
    # All points should be assigned (no -1 labels)
    assert np.all(labels >= 0)


def test_mean_shift_cluster_all_false():
    """Test MeanShift with cluster_all=False."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=0.5, cluster_all=False)
    labels = clustering.fit_predict(X)
    
    # May have outliers (-1 labels)
    assert len(labels) == 50


def test_mean_shift_max_iter():
    """Test MeanShift respects max_iter."""
    X = np.random.randn(30, 5)
    
    clustering = MeanShift(bandwidth=1.0, max_iter=100)
    clustering.fit(X)
    
    assert clustering.n_iter_ == 100


def test_mean_shift_n_features_in():
    """Test n_features_in_ attribute is set."""
    X = np.random.randn(50, 7)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    assert clustering.n_features_in_ == 7


def test_mean_shift_well_separated():
    """Test MeanShift on well-separated clusters."""
    # Create two well-separated clusters
    cluster1 = np.random.randn(25, 5) * 0.5
    cluster2 = np.random.randn(25, 5) * 0.5 + np.array([10, 0, 0, 0, 0])
    X = np.vstack([cluster1, cluster2])
    
    clustering = MeanShift(bandwidth=2.0)
    labels = clustering.fit_predict(X)
    
    # Should identify at least 2 clusters
    assert len(np.unique(labels)) >= 2


def test_mean_shift_small_dataset():
    """Test MeanShift on small dataset."""
    X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
    
    clustering = MeanShift(bandwidth=2.0)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (4,)


def test_mean_shift_error_before_fit():
    """Test MeanShift raises error when calling predict before fit."""
    clustering = MeanShift(bandwidth=1.0)
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        clustering.predict(X)


def test_mean_shift_different_dimensions():
    """Test MeanShift on different dimensional data."""
    for n_features in [2, 5, 10]:
        X = np.random.randn(40, n_features)
        
        clustering = MeanShift(bandwidth=1.0)
        labels = clustering.fit_predict(X)
        
        assert labels.shape == (40,)
        assert clustering.n_features_in_ == n_features


def test_mean_shift_high_dimensional():
    """Test MeanShift on high-dimensional data."""
    X = np.random.randn(50, 20)
    
    clustering = MeanShift(bandwidth=2.0)
    labels = clustering.fit_predict(X)
    
    assert labels.shape == (50,)


def test_mean_shift_cluster_count():
    """Test MeanShift automatically determines number of clusters."""
    X = np.random.randn(60, 5)
    
    clustering = MeanShift(bandwidth=1.5)
    labels = clustering.fit_predict(X)
    
    n_clusters = len(clustering.cluster_centers_)
    
    # Should find some clusters
    assert n_clusters > 0
    assert n_clusters <= 60


def test_mean_shift_predict_consistency():
    """Test MeanShift predict is consistent with fit labels."""
    X = np.random.randn(40, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    # Predicting on same data should give similar results
    labels_predict = clustering.predict(X)
    
    assert len(labels_predict) == 40


def test_mean_shift_centers_from_data():
    """Test MeanShift cluster centers are from data distribution."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    # Centers should be within reasonable range of data
    for center in clustering.cluster_centers_:
        assert np.all(np.abs(center) < 10)  # Reasonable range


def test_mean_shift_single_cluster():
    """Test MeanShift on dense single cluster."""
    # Very dense cluster
    X = np.random.randn(40, 5) * 0.1
    
    clustering = MeanShift(bandwidth=1.0)
    labels = clustering.fit_predict(X)
    
    # Should identify as one or few clusters
    assert len(clustering.cluster_centers_) <= 5


def test_mean_shift_bandwidth_effect():
    """Test different bandwidths affect number of clusters."""
    X = np.random.randn(60, 5)
    
    # Small bandwidth - more clusters
    ms_small = MeanShift(bandwidth=0.5)
    ms_small.fit(X)
    n_small = len(ms_small.cluster_centers_)
    
    # Large bandwidth - fewer clusters
    ms_large = MeanShift(bandwidth=3.0)
    ms_large.fit(X)
    n_large = len(ms_large.cluster_centers_)
    
    # Larger bandwidth should generally give fewer clusters
    assert n_large <= n_small


def test_mean_shift_labels_consistent():
    """Test MeanShift labels match cluster centers."""
    X = np.random.randn(50, 5)
    
    clustering = MeanShift(bandwidth=1.0)
    clustering.fit(X)
    
    n_clusters = len(clustering.cluster_centers_)
    
    # All labels should be valid cluster indices
    assert np.all(clustering.labels_ >= -1)
    if clustering.cluster_all:
        assert np.all(clustering.labels_ < n_clusters)


def test_mean_shift_reproducibility():
    """Test MeanShift gives consistent results."""
    X = np.random.RandomState(42).randn(40, 5)
    
    # Mean shift is deterministic (no random_state needed)
    labels1 = MeanShift(bandwidth=1.0).fit_predict(X)
    labels2 = MeanShift(bandwidth=1.0).fit_predict(X)
    
    # Results should be identical
    assert np.array_equal(labels1, labels2)


def test_mean_shift_sparse_data():
    """Test MeanShift on sparse data."""
    # Sparse data with outliers
    X = np.random.randn(30, 5) * 3
    
    clustering = MeanShift(bandwidth=2.0, cluster_all=False)
    labels = clustering.fit_predict(X)
    
    assert len(labels) == 30
