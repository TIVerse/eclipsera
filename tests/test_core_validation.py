"""Tests for core validation utilities."""
import numpy as np
import pytest

from eclipsera.core.exceptions import ValidationError
from eclipsera.core.validation import (
    check_array,
    check_classification_targets,
    check_consistent_length,
    check_random_state,
    check_regression_targets,
    check_symmetric,
    check_X_y,
    column_or_1d,
    has_fit_parameter,
    type_of_target,
)


def test_check_array_valid():
    """Test check_array with valid input."""
    X = np.array([[1, 2], [3, 4]])
    X_checked = check_array(X)
    
    assert isinstance(X_checked, np.ndarray)
    assert X_checked.shape == (2, 2)


def test_check_array_list():
    """Test check_array with list input."""
    X = [[1, 2], [3, 4]]
    X_checked = check_array(X)
    
    assert isinstance(X_checked, np.ndarray)


def test_check_array_1d_raises():
    """Test check_array raises error for 1D array when 2D expected."""
    X = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Expected 2D array"):
        check_array(X, ensure_2d=True)


def test_check_array_nan_raises():
    """Test check_array raises error for NaN values."""
    X = np.array([[1, 2], [3, np.nan]])
    
    with pytest.raises(ValueError, match="NaN or infinity"):
        check_array(X, force_all_finite=True)


def test_check_array_allow_nan():
    """Test check_array allows NaN when specified."""
    X = np.array([[1, 2], [3, np.nan]])
    
    X_checked = check_array(X, force_all_finite='allow-nan')
    assert np.isnan(X_checked[1, 1])


def test_check_array_inf_raises():
    """Test check_array raises error for infinite values."""
    X = np.array([[1, 2], [3, np.inf]])
    
    with pytest.raises(ValueError, match="infinity"):
        check_array(X, force_all_finite=True)


def test_check_array_copy():
    """Test check_array copy behavior."""
    X = np.array([[1, 2], [3, 4]])
    X_checked = check_array(X, copy=True)
    
    X_checked[0, 0] = 999
    assert X[0, 0] == 1  # Original unchanged


def test_check_X_y():
    """Test check_X_y with valid inputs."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    X_checked, y_checked = check_X_y(X, y)
    
    assert X_checked.shape == (3, 2)
    assert y_checked.shape == (3,)


def test_check_X_y_inconsistent_length():
    """Test check_X_y raises error for inconsistent lengths."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1, 2])  # Wrong length
    
    with pytest.raises(ValueError, match="inconsistent"):
        check_X_y(X, y)


def test_check_X_y_multioutput():
    """Test check_X_y with multi-output y."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0, 1], [1, 0]])
    
    X_checked, y_checked = check_X_y(X, y, multi_output=True)
    
    assert y_checked.shape == (2, 2)


def test_check_consistent_length():
    """Test check_consistent_length."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Should not raise
    check_consistent_length(a, b)


def test_check_consistent_length_raises():
    """Test check_consistent_length raises error."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
    with pytest.raises(ValueError, match="inconsistent"):
        check_consistent_length(a, b)


def test_column_or_1d():
    """Test column_or_1d with 1D array."""
    y = np.array([1, 2, 3])
    y_checked = column_or_1d(y)
    
    assert y_checked.shape == (3,)


def test_column_or_1d_column():
    """Test column_or_1d with column vector."""
    y = np.array([[1], [2], [3]])
    y_checked = column_or_1d(y)
    
    assert y_checked.shape == (3,)


def test_column_or_1d_raises():
    """Test column_or_1d raises error for 2D array."""
    y = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="1d array"):
        column_or_1d(y)


def test_check_random_state_none():
    """Test check_random_state with None."""
    rs = check_random_state(None)
    assert rs is not None


def test_check_random_state_int():
    """Test check_random_state with int seed."""
    rs = check_random_state(42)
    assert rs is not None
    
    # Should produce reproducible results
    val1 = rs.rand()
    rs2 = check_random_state(42)
    val2 = rs2.rand()
    assert val1 == val2


def test_check_random_state_instance():
    """Test check_random_state with RandomState instance."""
    rs_orig = np.random.RandomState(42)
    rs = check_random_state(rs_orig)
    assert rs is rs_orig


def test_type_of_target_binary():
    """Test type_of_target for binary classification."""
    y = np.array([0, 1, 0, 1])
    assert type_of_target(y) == "binary"


def test_type_of_target_multiclass():
    """Test type_of_target for multiclass classification."""
    y = np.array([0, 1, 2, 1, 0])
    assert type_of_target(y) == "multiclass"


def test_type_of_target_continuous():
    """Test type_of_target for continuous regression."""
    y = np.array([0.1, 0.5, 0.9, 1.2])
    assert type_of_target(y) == "continuous"


def test_type_of_target_multioutput():
    """Test type_of_target for multioutput regression."""
    y = np.array([[0.1, 0.2], [0.5, 0.6]])
    assert type_of_target(y) == "continuous-multioutput"


def test_check_classification_targets():
    """Test check_classification_targets accepts valid targets."""
    y = np.array([0, 1, 2, 1, 0])
    # Should not raise
    check_classification_targets(y)


def test_check_classification_targets_continuous_raises():
    """Test check_classification_targets raises for continuous."""
    y = np.array([0.1, 0.5, 0.9])
    
    with pytest.raises(ValueError, match="Unknown label type"):
        check_classification_targets(y)


def test_check_regression_targets():
    """Test check_regression_targets accepts valid targets."""
    y = np.array([0.1, 0.5, 0.9])
    # Should not raise
    check_regression_targets(y)


def test_check_regression_targets_discrete_raises():
    """Test check_regression_targets raises for discrete."""
    y = np.array([0, 1, 2])
    
    with pytest.raises(ValueError, match="Unknown label type"):
        check_regression_targets(y)


def test_check_symmetric():
    """Test check_symmetric with symmetric matrix."""
    X = np.array([[1, 2], [2, 1]])
    # Should not raise
    assert check_symmetric(X)


def test_check_symmetric_asymmetric_raises():
    """Test check_symmetric raises for asymmetric matrix."""
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="not symmetric"):
        check_symmetric(X)


def test_check_symmetric_non_square_raises():
    """Test check_symmetric raises for non-square matrix."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    with pytest.raises(ValueError, match="must be square"):
        check_symmetric(X)


def test_has_fit_parameter():
    """Test has_fit_parameter."""
    from eclipsera.ml.linear import LinearRegression
    
    model = LinearRegression()
    assert has_fit_parameter(model, "X")
    assert has_fit_parameter(model, "y")
    assert not has_fit_parameter(model, "nonexistent")


class DummyEstimatorNoFit:
    """Dummy estimator without fit method."""
    pass


def test_has_fit_parameter_no_fit():
    """Test has_fit_parameter with estimator lacking fit method."""
    est = DummyEstimatorNoFit()
    assert not has_fit_parameter(est, "X")
