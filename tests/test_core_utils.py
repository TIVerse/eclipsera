"""Tests for core utility functions."""
import numpy as np
import pytest
from numpy.random import RandomState

from eclipsera.core.utils import (
    check_random_state,
    check_scalar,
    compute_sample_weight,
    log_softmax,
    resample,
    safe_indexing,
    safe_sparse_dot,
    safe_sqr,
    shuffle,
    softmax,
)


def test_safe_sparse_dot_dense():
    """Test safe_sparse_dot with dense arrays."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    result = safe_sparse_dot(a, b)
    expected = np.dot(a, b)
    
    assert np.allclose(result, expected)


def test_safe_sparse_dot_vector():
    """Test safe_sparse_dot with matrix and vector."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    
    result = safe_sparse_dot(a, b)
    expected = np.dot(a, b)
    
    assert np.allclose(result, expected)


def test_check_random_state_int():
    """Test check_random_state with integer seed."""
    rs = check_random_state(42)
    assert isinstance(rs, RandomState)
    
    # Test reproducibility
    rs1 = check_random_state(42)
    rs2 = check_random_state(42)
    assert rs1.rand() == rs2.rand()


def test_check_random_state_none():
    """Test check_random_state with None."""
    rs = check_random_state(None)
    assert rs is not None


def test_safe_indexing_array():
    """Test safe_indexing with numpy array."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 2])
    
    result = safe_indexing(X, indices)
    expected = np.array([[1, 2], [5, 6]])
    
    assert np.array_equal(result, expected)


def test_safe_indexing_list():
    """Test safe_indexing with list."""
    X = [1, 2, 3, 4, 5]
    indices = np.array([0, 2, 4])
    
    result = safe_indexing(X, indices)
    expected = [1, 3, 5]
    
    assert result == expected


def test_shuffle_array():
    """Test shuffle with numpy array."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    X_shuffled = shuffle(X, random_state=42)
    
    # Shape should be preserved
    assert X_shuffled.shape == X.shape
    
    # Elements should be the same (just reordered)
    assert set(map(tuple, X)) == set(map(tuple, X_shuffled))


def test_shuffle_multiple_arrays():
    """Test shuffle with multiple arrays."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])
    
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    
    assert X_shuffled.shape == X.shape
    assert y_shuffled.shape == y.shape


def test_shuffle_n_samples():
    """Test shuffle with n_samples parameter."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    X_shuffled = shuffle(X, random_state=42, n_samples=2)
    
    assert X_shuffled.shape == (2, 2)


def test_resample_with_replacement():
    """Test resample with replacement."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    X_resampled = resample(X, replace=True, n_samples=5, random_state=42)
    
    assert X_resampled.shape == (5, 2)


def test_resample_without_replacement():
    """Test resample without replacement."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    X_resampled = resample(X, replace=False, n_samples=2, random_state=42)
    
    assert X_resampled.shape == (2, 2)


def test_resample_stratified():
    """Test stratified resampling."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    
    X_resampled, y_resampled = resample(
        X, y, stratify=y, n_samples=4, random_state=42
    )
    
    assert X_resampled.shape == (4, 2)
    assert y_resampled.shape == (4,)


def test_compute_sample_weight_none():
    """Test compute_sample_weight with None."""
    y = np.array([0, 1, 0, 1])
    
    weights = compute_sample_weight(None, y)
    
    assert np.array_equal(weights, np.ones(4))


def test_compute_sample_weight_balanced():
    """Test compute_sample_weight with balanced."""
    y = np.array([0, 0, 0, 1])
    
    weights = compute_sample_weight("balanced", y)
    
    # Class 0 has 3 samples, class 1 has 1 sample
    # Weight for class 0: 4 / (2 * 3) = 2/3
    # Weight for class 1: 4 / (2 * 1) = 2
    assert weights[0] < weights[3]


def test_compute_sample_weight_dict():
    """Test compute_sample_weight with custom dict."""
    y = np.array([0, 1, 0, 1])
    class_weight = {0: 2.0, 1: 1.0}
    
    weights = compute_sample_weight(class_weight, y)
    
    assert weights[0] == 2.0
    assert weights[1] == 1.0


def test_softmax():
    """Test softmax function."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    probs = softmax(X)
    
    # Each row should sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # All probabilities should be positive
    assert np.all(probs > 0)


def test_softmax_numerical_stability():
    """Test softmax with large values."""
    X = np.array([[1000, 1001, 1002]])
    
    probs = softmax(X)
    
    # Should not overflow
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(), 1.0)


def test_log_softmax():
    """Test log_softmax function."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    log_probs = log_softmax(X)
    
    # Log probabilities should be negative
    assert np.all(log_probs <= 0)
    
    # Should match log of softmax
    probs = softmax(X.copy())
    expected = np.log(probs)
    
    assert np.allclose(log_probs, expected, atol=1e-10)


def test_safe_sqr():
    """Test safe_sqr function."""
    X = np.array([1, 2, 3, 4])
    
    X_sqr = safe_sqr(X)
    expected = np.array([1, 4, 9, 16])
    
    assert np.array_equal(X_sqr, expected)


def test_safe_sqr_copy():
    """Test safe_sqr with copy parameter."""
    X = np.array([1, 2, 3])
    
    X_sqr = safe_sqr(X, copy=True)
    X_sqr[0] = 999
    
    # Original should be unchanged
    assert X[0] == 1


def test_check_scalar_valid():
    """Test check_scalar with valid input."""
    # Should not raise
    check_scalar(5, "param", int)
    check_scalar(5.0, "param", float)
    check_scalar(5, "param", (int, float))


def test_check_scalar_invalid_type():
    """Test check_scalar with invalid type."""
    with pytest.raises(TypeError):
        check_scalar("5", "param", int)


def test_check_scalar_min_val():
    """Test check_scalar with minimum value."""
    # Should not raise
    check_scalar(5, "param", int, min_val=0)
    
    with pytest.raises(ValueError, match="must be >="):
        check_scalar(-1, "param", int, min_val=0)


def test_check_scalar_max_val():
    """Test check_scalar with maximum value."""
    # Should not raise
    check_scalar(5, "param", int, max_val=10)
    
    with pytest.raises(ValueError, match="must be <="):
        check_scalar(15, "param", int, max_val=10)


def test_check_scalar_boundaries():
    """Test check_scalar with different boundary options."""
    # Test exclusive boundaries
    with pytest.raises(ValueError):
        check_scalar(0, "param", int, min_val=0, include_boundaries="neither")
    
    with pytest.raises(ValueError):
        check_scalar(10, "param", int, max_val=10, include_boundaries="neither")
    
    # Should not raise with exclusive left boundary
    check_scalar(1, "param", int, min_val=0, include_boundaries="right")
