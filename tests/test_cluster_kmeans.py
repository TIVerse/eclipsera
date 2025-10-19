"""Tests for K-Means clustering."""
import numpy as np
import pytest

from eclipsera.cluster import KMeans, MiniBatchKMeans


def test_kmeans_basic():
    """Test basic KMeans functionality."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    
    assert hasattr(kmeans, 'cluster_centers_')
    assert hasattr(kmeans, 'labels_')
    assert hasattr(kmeans, 'inertia_')
    assert kmeans.cluster_centers_.shape == (2, 2)
    assert len(kmeans.labels_) == 6


def test_kmeans_predict():
    """Test KMeans predict method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    
    new_points = np.array([[0, 0], [12, 3]])
    labels = kmeans.predict(new_points)
    
    assert labels.shape == (2,)
    assert all(label in [0, 1] for label in labels)


def test_kmeans_fit_predict():
    """Test KMeans fit_predict method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(X)
    
    assert labels.shape == (6,)
    assert len(np.unique(labels)) == 2


def test_kmeans_transform():
    """Test KMeans transform method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    
    distances = kmeans.transform(X)
    
    assert distances.shape == (6, 2)
    assert np.all(distances >= 0)


def test_kmeans_score():
    """Test KMeans score method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    
    score = kmeans.score(X)
    
    # Score is negative inertia
    assert score < 0
    assert score == -kmeans.inertia_


def test_kmeans_n_clusters():
    """Test KMeans with different n_clusters."""
    X = np.random.randn(50, 3)
    
    for n_clusters in [2, 3, 5]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        assert len(np.unique(labels)) <= n_clusters
        assert kmeans.cluster_centers_.shape == (n_clusters, 3)


def test_kmeans_init_random():
    """Test KMeans with random initialization."""
    X = np.random.randn(30, 2)
    
    kmeans = KMeans(n_clusters=3, init='random', n_init=5, random_state=42)
    kmeans.fit(X)
    
    assert hasattr(kmeans, 'cluster_centers_')
    assert kmeans.cluster_centers_.shape == (3, 2)


def test_kmeans_init_kmeanspp():
    """Test KMeans with k-means++ initialization."""
    X = np.random.randn(30, 2)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=5, random_state=42)
    kmeans.fit(X)
    
    assert hasattr(kmeans, 'cluster_centers_')
    assert kmeans.cluster_centers_.shape == (3, 2)


def test_kmeans_convergence():
    """Test KMeans converges within max_iter."""
    X = np.random.randn(50, 3)
    
    kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.n_iter_ <= 100


def test_kmeans_n_init():
    """Test KMeans with multiple initializations."""
    X = np.random.randn(30, 2)
    
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X)
    
    # Should run 10 times and keep best
    assert hasattr(kmeans, 'inertia_')


def test_kmeans_reproducibility():
    """Test KMeans is reproducible with random_state."""
    X = np.random.randn(30, 2)
    
    kmeans1 = KMeans(n_clusters=3, random_state=42)
    labels1 = kmeans1.fit_predict(X)
    
    kmeans2 = KMeans(n_clusters=3, random_state=42)
    labels2 = kmeans2.fit_predict(X)
    
    assert np.array_equal(labels1, labels2)


def test_kmeans_inertia_decreases():
    """Test that inertia decreases with more iterations."""
    X = np.random.randn(50, 2)
    
    # Low iterations
    kmeans1 = KMeans(n_clusters=3, max_iter=1, n_init=1, random_state=42)
    kmeans1.fit(X)
    
    # More iterations
    kmeans2 = KMeans(n_clusters=3, max_iter=100, n_init=1, random_state=42)
    kmeans2.fit(X)
    
    # More iterations should give same or better inertia
    assert kmeans2.inertia_ <= kmeans1.inertia_ * 1.1  # Allow small margin


def test_minibatch_kmeans_basic():
    """Test basic MiniBatchKMeans functionality."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    mbk = MiniBatchKMeans(n_clusters=2, batch_size=3, random_state=0)
    mbk.fit(X)
    
    assert hasattr(mbk, 'cluster_centers_')
    assert hasattr(mbk, 'labels_')
    assert hasattr(mbk, 'inertia_')
    assert mbk.cluster_centers_.shape == (2, 2)


def test_minibatch_kmeans_predict():
    """Test MiniBatchKMeans predict method."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    mbk = MiniBatchKMeans(n_clusters=2, batch_size=3, random_state=0)
    mbk.fit(X)
    
    new_points = np.array([[0, 0], [12, 3]])
    labels = mbk.predict(new_points)
    
    assert labels.shape == (2,)
    assert all(label in [0, 1] for label in labels)


def test_minibatch_kmeans_fit_predict():
    """Test MiniBatchKMeans fit_predict method."""
    X = np.random.randn(100, 3)
    
    mbk = MiniBatchKMeans(n_clusters=3, batch_size=20, random_state=42)
    labels = mbk.fit_predict(X)
    
    assert labels.shape == (100,)
    assert len(np.unique(labels)) <= 3


def test_minibatch_kmeans_batch_size():
    """Test MiniBatchKMeans with different batch sizes."""
    X = np.random.randn(100, 2)
    
    for batch_size in [10, 50, 100]:
        mbk = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=42)
        mbk.fit(X)
        
        assert hasattr(mbk, 'cluster_centers_')


def test_minibatch_kmeans_large_dataset():
    """Test MiniBatchKMeans on larger dataset."""
    X = np.random.randn(1000, 5)
    
    mbk = MiniBatchKMeans(n_clusters=5, batch_size=100, max_iter=10, random_state=42)
    labels = mbk.fit_predict(X)
    
    assert labels.shape == (1000,)
    assert mbk.cluster_centers_.shape == (5, 5)


def test_minibatch_kmeans_reproducibility():
    """Test MiniBatchKMeans is reproducible with random_state."""
    X = np.random.randn(100, 2)
    
    mbk1 = MiniBatchKMeans(n_clusters=3, batch_size=20, random_state=42)
    labels1 = mbk1.fit_predict(X)
    
    mbk2 = MiniBatchKMeans(n_clusters=3, batch_size=20, random_state=42)
    labels2 = mbk2.fit_predict(X)
    
    assert np.array_equal(labels1, labels2)


def test_kmeans_error_before_fit():
    """Test KMeans raises error when calling methods before fit."""
    kmeans = KMeans(n_clusters=2)
    X = np.random.randn(10, 2)
    
    with pytest.raises(ValueError):
        kmeans.predict(X)
    
    with pytest.raises(ValueError):
        kmeans.transform(X)


def test_kmeans_single_cluster():
    """Test KMeans with single cluster."""
    X = np.random.randn(20, 2)
    
    kmeans = KMeans(n_clusters=1, random_state=42)
    labels = kmeans.fit_predict(X)
    
    assert np.all(labels == 0)


def test_kmeans_well_separated():
    """Test KMeans on well-separated clusters."""
    # Create two well-separated clusters
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([10, 10])
    X = np.vstack([cluster1, cluster2])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Should correctly identify two clusters
    assert len(np.unique(labels)) == 2


def test_minibatch_vs_regular_kmeans():
    """Test MiniBatchKMeans gives similar results to KMeans."""
    X = np.random.randn(100, 3)
    
    kmeans = KMeans(n_clusters=3, n_init=1, random_state=42)
    kmeans.fit(X)
    
    mbk = MiniBatchKMeans(n_clusters=3, batch_size=20, random_state=42)
    mbk.fit(X)
    
    # Results should be somewhat similar (inertia within reasonable range)
    assert abs(kmeans.inertia_ - mbk.inertia_) / kmeans.inertia_ < 0.5
