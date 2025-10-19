"""Tests for core base classes."""
import numpy as np
import pytest

from eclipsera.core.base import (
    BaseClassifier,
    BaseEstimator,
    BaseRegressor,
    clone,
)


class DummyEstimator(BaseEstimator):
    """Dummy estimator for testing."""
    
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y=None):
        self.fitted_value_ = self.param1 + self.param2
        return self


def test_get_params():
    """Test get_params method."""
    est = DummyEstimator(param1=10, param2=20)
    params = est.get_params()
    
    assert params == {"param1": 10, "param2": 20}


def test_set_params():
    """Test set_params method."""
    est = DummyEstimator()
    est.set_params(param1=100)
    
    assert est.param1 == 100
    assert est.param2 == 2  # unchanged


def test_set_params_invalid():
    """Test set_params with invalid parameter."""
    est = DummyEstimator()
    
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(invalid_param=5)


def test_repr():
    """Test __repr__ method."""
    est = DummyEstimator(param1=5, param2=10)
    repr_str = repr(est)
    
    assert "DummyEstimator" in repr_str
    assert "param1=5" in repr_str
    assert "param2=10" in repr_str


def test_clone():
    """Test clone function."""
    est = DummyEstimator(param1=42)
    est.fit(np.array([[1, 2]]), np.array([1]))
    
    est_clone = clone(est)
    
    # Parameters should be copied
    assert est_clone.param1 == 42
    
    # Fitted attributes should NOT be copied
    assert not hasattr(est_clone, "fitted_value_")


def test_clone_unfitted():
    """Test cloning unfitted estimator."""
    est = DummyEstimator(param1=5, param2=10)
    est_clone = clone(est)
    
    assert est_clone.param1 == 5
    assert est_clone.param2 == 10
    assert est_clone is not est


def test_get_param_names():
    """Test _get_param_names class method."""
    param_names = DummyEstimator._get_param_names()
    
    assert "param1" in param_names
    assert "param2" in param_names
    assert "self" not in param_names
