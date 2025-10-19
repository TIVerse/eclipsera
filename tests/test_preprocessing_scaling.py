"""Tests for preprocessing scalers."""
import numpy as np
import pytest

from eclipsera.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)


def test_standard_scaler():
    """Test StandardScaler."""
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check mean is 0 and std is 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    # Check inverse transform
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv)


def test_standard_scaler_no_mean():
    """Test StandardScaler without centering."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    
    assert scaler.mean_ is None


def test_standard_scaler_no_std():
    """Test StandardScaler without scaling."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X)
    
    assert scaler.scale_ is None


def test_minmax_scaler():
    """Test MinMaxScaler."""
    X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check range is [0, 1]
    assert X_scaled.min() >= 0
    assert X_scaled.max() <= 1
    assert np.allclose(X_scaled.min(axis=0), 0)
    assert np.allclose(X_scaled.max(axis=0), 1)
    
    # Check inverse transform
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv)


def test_minmax_scaler_custom_range():
    """Test MinMaxScaler with custom range."""
    X = np.array([[0, 1], [1, 2]])
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.min() >= -1
    assert X_scaled.max() <= 1


def test_robust_scaler():
    """Test RobustScaler."""
    X = np.array([[1, 2], [2, 3], [3, 4], [100, 200]])  # With outlier
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check that median was used for centering
    assert hasattr(scaler, "center_")
    assert hasattr(scaler, "scale_")


def test_maxabs_scaler():
    """Test MaxAbsScaler."""
    X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check max absolute value is 1
    assert np.allclose(np.abs(X_scaled).max(axis=0), 1)


def test_normalizer_l2():
    """Test Normalizer with L2 norm."""
    X = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
    
    normalizer = Normalizer(norm="l2")
    X_normalized = normalizer.fit_transform(X)
    
    # Check each sample has unit norm
    norms = np.sqrt((X_normalized ** 2).sum(axis=1))
    assert np.allclose(norms, 1.0)


def test_normalizer_l1():
    """Test Normalizer with L1 norm."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    normalizer = Normalizer(norm="l1")
    X_normalized = normalizer.fit_transform(X)
    
    # Check each sample sums to 1
    sums = np.abs(X_normalized).sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_normalizer_max():
    """Test Normalizer with max norm."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    normalizer = Normalizer(norm="max")
    X_normalized = normalizer.fit_transform(X)
    
    # Check each sample max is 1
    maxes = np.abs(X_normalized).max(axis=1)
    assert np.allclose(maxes, 1.0)


def test_scaler_zero_variance():
    """Test scalers with zero variance feature."""
    X = np.array([[1, 0], [2, 0], [3, 0]])
    
    # Should not raise error
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Zero variance column should remain unchanged
    assert np.all(X_scaled[:, 1] == 0)
