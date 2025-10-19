"""Pytest configuration and fixtures for Eclipsera tests."""
import numpy as np
import pytest


@pytest.fixture
def random_state():
    """Fixed random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def classification_data(random_state):
    """Generate simple classification dataset."""
    n_samples = 100
    n_features = 5
    n_classes = 3
    
    X = random_state.randn(n_samples, n_features)
    y = random_state.randint(0, n_classes, size=n_samples)
    
    return X, y


@pytest.fixture
def binary_classification_data(random_state):
    """Generate binary classification dataset."""
    n_samples = 100
    n_features = 5
    
    X = random_state.randn(n_samples, n_features)
    y = random_state.randint(0, 2, size=n_samples)
    
    return X, y


@pytest.fixture
def regression_data(random_state):
    """Generate simple regression dataset."""
    n_samples = 100
    n_features = 5
    
    X = random_state.randn(n_samples, n_features)
    true_coef = random_state.randn(n_features)
    y = X @ true_coef + random_state.randn(n_samples) * 0.1
    
    return X, y


@pytest.fixture
def multioutput_regression_data(random_state):
    """Generate multioutput regression dataset."""
    n_samples = 100
    n_features = 5
    n_outputs = 3
    
    X = random_state.randn(n_samples, n_features)
    true_coef = random_state.randn(n_features, n_outputs)
    y = X @ true_coef + random_state.randn(n_samples, n_outputs) * 0.1
    
    return X, y
