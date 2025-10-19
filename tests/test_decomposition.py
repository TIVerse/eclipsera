"""Tests for dimensionality reduction."""
import numpy as np
import pytest

from eclipsera.decomposition import PCA, TruncatedSVD


def test_pca_basic():
    """Test basic PCA functionality."""
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    
    pca = PCA(n_components=2)
    pca.fit(X)
    
    assert hasattr(pca, 'components_')
    assert hasattr(pca, 'explained_variance_')
    assert hasattr(pca, 'explained_variance_ratio_')
    assert hasattr(pca, 'mean_')
    assert pca.components_.shape == (2, 2)


def test_pca_transform():
    """Test PCA transform method."""
    X = np.random.randn(20, 5)
    
    pca = PCA(n_components=3)
    pca.fit(X)
    X_transformed = pca.transform(X)
    
    assert X_transformed.shape == (20, 3)


def test_pca_fit_transform():
    """Test PCA fit_transform method."""
    X = np.random.randn(20, 5)
    
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)
    
    assert X_transformed.shape == (20, 3)
    assert hasattr(pca, 'components_')


def test_pca_inverse_transform():
    """Test PCA inverse_transform method."""
    X = np.random.randn(20, 5)
    
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    assert X_reconstructed.shape == X.shape


def test_pca_reconstruction():
    """Test PCA reconstruction error."""
    X = np.random.randn(20, 5)
    
    pca = PCA(n_components=5)  # Keep all components
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # With all components, reconstruction should be nearly perfect
    assert np.allclose(X, X_reconstructed, atol=1e-10)


def test_pca_explained_variance():
    """Test PCA explained variance sums to 1."""
    X = np.random.randn(50, 10)
    
    pca = PCA()
    pca.fit(X)
    
    # Total variance should be close to 1
    total_var = pca.explained_variance_ratio_.sum()
    assert 0.99 <= total_var <= 1.01


def test_pca_n_components_int():
    """Test PCA with integer n_components."""
    X = np.random.randn(20, 10)
    
    pca = PCA(n_components=5)
    X_transformed = pca.fit_transform(X)
    
    assert X_transformed.shape == (20, 5)
    assert pca.components_.shape == (5, 10)


def test_pca_n_components_float():
    """Test PCA with float n_components (variance ratio)."""
    X = np.random.randn(50, 20)
    
    pca = PCA(n_components=0.95)  # Keep 95% variance
    pca.fit(X)
    
    # Should automatically select number of components
    assert pca.n_components_ <= 20
    assert pca.explained_variance_ratio_.sum() >= 0.95


def test_pca_n_components_none():
    """Test PCA with n_components=None (keep all)."""
    X = np.random.randn(20, 10)
    
    pca = PCA(n_components=None)
    pca.fit(X)
    
    # Should keep min(n_samples, n_features)
    assert pca.n_components_ == min(20, 10)


def test_pca_whitening():
    """Test PCA with whitening."""
    X = np.random.randn(50, 10)
    
    pca = PCA(n_components=5, whiten=True)
    X_transformed = pca.fit_transform(X)
    
    # With whitening, transformed data should have unit variance
    variances = np.var(X_transformed, axis=0)
    assert np.allclose(variances, 1.0, atol=0.1)


def test_pca_centering():
    """Test PCA centers the data."""
    X = np.random.randn(50, 5) + 10  # Offset data
    
    pca = PCA(n_components=3)
    pca.fit(X)
    
    # Mean should be non-zero
    assert not np.allclose(pca.mean_, 0)
    
    # Transformed data should be centered
    X_transformed = pca.transform(X)
    assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)


def test_pca_deterministic():
    """Test PCA is deterministic."""
    X = np.random.RandomState(42).randn(30, 5)
    
    pca1 = PCA(n_components=3)
    X1 = pca1.fit_transform(X)
    
    pca2 = PCA(n_components=3)
    X2 = pca2.fit_transform(X)
    
    # Results should be identical (up to sign)
    assert np.allclose(np.abs(X1), np.abs(X2))


def test_pca_error_before_fit():
    """Test PCA raises error when calling methods before fit."""
    pca = PCA(n_components=2)
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        pca.transform(X)
    
    with pytest.raises(ValueError):
        pca.inverse_transform(X)


def test_pca_variance_ordering():
    """Test PCA components are ordered by variance."""
    X = np.random.randn(50, 10)
    
    pca = PCA()
    pca.fit(X)
    
    # Explained variance should be in descending order
    assert np.all(np.diff(pca.explained_variance_) <= 0)


def test_truncated_svd_basic():
    """Test basic TruncatedSVD functionality."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    svd = TruncatedSVD(n_components=2)
    svd.fit(X)
    
    assert hasattr(svd, 'components_')
    assert hasattr(svd, 'explained_variance_')
    assert hasattr(svd, 'singular_values_')
    assert svd.components_.shape == (2, 3)


def test_truncated_svd_transform():
    """Test TruncatedSVD transform method."""
    X = np.random.randn(20, 10)
    
    svd = TruncatedSVD(n_components=5)
    svd.fit(X)
    X_transformed = svd.transform(X)
    
    assert X_transformed.shape == (20, 5)


def test_truncated_svd_fit_transform():
    """Test TruncatedSVD fit_transform method."""
    X = np.random.randn(20, 10)
    
    svd = TruncatedSVD(n_components=5)
    X_transformed = svd.fit_transform(X)
    
    assert X_transformed.shape == (20, 5)
    assert hasattr(svd, 'components_')


def test_truncated_svd_inverse_transform():
    """Test TruncatedSVD inverse_transform method."""
    X = np.random.randn(20, 10)
    
    svd = TruncatedSVD(n_components=5)
    X_transformed = svd.fit_transform(X)
    X_reconstructed = svd.inverse_transform(X_transformed)
    
    assert X_reconstructed.shape == X.shape


def test_truncated_svd_no_centering():
    """Test TruncatedSVD does not center data."""
    X = np.random.randn(50, 10) + 5  # Offset data
    
    svd = TruncatedSVD(n_components=5)
    X_transformed = svd.fit_transform(X)
    
    # Unlike PCA, transformed data is NOT centered
    assert not np.allclose(X_transformed.mean(axis=0), 0, atol=0.1)


def test_truncated_svd_n_components():
    """Test TruncatedSVD with different n_components."""
    X = np.random.randn(30, 10)
    
    for n_comp in [2, 5, 8]:
        svd = TruncatedSVD(n_components=n_comp)
        X_transformed = svd.fit_transform(X)
        
        assert X_transformed.shape == (30, n_comp)
        assert svd.components_.shape == (n_comp, 10)


def test_truncated_svd_error_too_many_components():
    """Test TruncatedSVD raises error if n_components too large."""
    X = np.random.randn(10, 5)
    
    svd = TruncatedSVD(n_components=10)  # More than min(n_samples, n_features)
    
    with pytest.raises(ValueError):
        svd.fit(X)


def test_truncated_svd_deterministic():
    """Test TruncatedSVD is deterministic."""
    X = np.random.RandomState(42).randn(30, 10)
    
    svd1 = TruncatedSVD(n_components=5)
    X1 = svd1.fit_transform(X)
    
    svd2 = TruncatedSVD(n_components=5)
    X2 = svd2.fit_transform(X)
    
    # Results should be identical (up to sign)
    assert np.allclose(np.abs(X1), np.abs(X2))


def test_truncated_svd_error_before_fit():
    """Test TruncatedSVD raises error before fit."""
    svd = TruncatedSVD(n_components=2)
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        svd.transform(X)
    
    with pytest.raises(ValueError):
        svd.inverse_transform(X)


def test_pca_vs_truncated_svd():
    """Test PCA and TruncatedSVD give different results due to centering."""
    X = np.random.randn(30, 5) + 5  # Offset data
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    svd = TruncatedSVD(n_components=3)
    X_svd = svd.fit_transform(X)
    
    # Results should be different because PCA centers
    assert not np.allclose(X_pca, X_svd)


def test_pca_in_pipeline():
    """Test PCA works in pipeline."""
    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    
    X = np.random.randn(50, 10)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5))
    ])
    
    X_transformed = pipe.fit_transform(X)
    assert X_transformed.shape == (50, 5)


def test_truncated_svd_explained_variance():
    """Test TruncatedSVD explained variance."""
    X = np.random.randn(50, 20)
    
    svd = TruncatedSVD(n_components=10)
    svd.fit(X)
    
    # Explained variance ratio should sum to less than 1 (unless all components)
    assert 0 < svd.explained_variance_ratio_.sum() <= 1


def test_pca_small_dataset():
    """Test PCA on small dataset."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)
    
    assert X_transformed.shape == (3, 1)


def test_truncated_svd_small_dataset():
    """Test TruncatedSVD on small dataset."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    svd = TruncatedSVD(n_components=1)
    X_transformed = svd.fit_transform(X)
    
    assert X_transformed.shape == (3, 1)
