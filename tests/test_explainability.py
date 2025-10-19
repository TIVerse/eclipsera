"""Tests for explainability functionality."""
import numpy as np
import pytest

from eclipsera.explainability import (
    permutation_importance,
    partial_dependence,
    get_feature_importance,
)
from eclipsera.ml import (
    RandomForestClassifier,
    LogisticRegression,
    RandomForestRegressor,
)


def test_permutation_importance_basic():
    """Test basic permutation importance."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    
    assert 'importances' in result
    assert 'importances_mean' in result
    assert 'importances_std' in result


def test_permutation_importance_shape():
    """Test permutation importance output shape."""
    X = np.random.randn(80, 12)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
    
    assert result['importances'].shape == (12, 10)
    assert result['importances_mean'].shape == (12,)
    assert result['importances_std'].shape == (12,)


def test_permutation_importance_repeats():
    """Test permutation importance with different n_repeats."""
    X = np.random.randn(60, 8)
    y = np.random.randint(0, 2, 60)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    for n_repeats in [3, 5, 10]:
        result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=42)
        assert result['importances'].shape[1] == n_repeats


def test_permutation_importance_random_state():
    """Test permutation importance reproducibility."""
    X = np.random.RandomState(42).randn(60, 8)
    y = np.random.RandomState(42).randint(0, 2, 60)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result1 = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    result2 = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    
    assert np.allclose(result1['importances_mean'], result2['importances_mean'])


def test_permutation_importance_regression():
    """Test permutation importance on regression."""
    X = np.random.randn(80, 10)
    y = np.random.randn(80)
    
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)
    
    result = permutation_importance(reg, X, y, n_repeats=5, random_state=42)
    
    assert result['importances_mean'].shape == (10,)


def test_permutation_importance_error_no_predict():
    """Test permutation importance error without predict method."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    class DummyEstimator:
        pass
    
    estimator = DummyEstimator()
    
    with pytest.raises(ValueError, match="predict method"):
        permutation_importance(estimator, X, y)


def test_partial_dependence_basic():
    """Test basic partial dependence."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = partial_dependence(clf, X, features=[0, 1])
    
    assert 'values' in result
    assert 'predictions' in result
    assert 'features' in result


def test_partial_dependence_single_feature():
    """Test partial dependence for single feature."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = partial_dependence(clf, X, features=[0])
    
    assert len(result['values']) == 1
    assert len(result['predictions']) == 1
    assert result['features'] == [0]


def test_partial_dependence_multiple_features():
    """Test partial dependence for multiple features."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = partial_dependence(clf, X, features=[0, 2, 5])
    
    assert len(result['values']) == 3
    assert len(result['predictions']) == 3
    assert result['features'] == [0, 2, 5]


def test_partial_dependence_grid_resolution():
    """Test partial dependence with different grid resolutions."""
    X = np.random.randn(60, 8)
    y = np.random.randint(0, 2, 60)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    for grid_resolution in [20, 50, 100]:
        result = partial_dependence(clf, X, features=[0], grid_resolution=grid_resolution)
        assert len(result['values'][0]) == grid_resolution
        assert len(result['predictions'][0]) == grid_resolution


def test_partial_dependence_regression():
    """Test partial dependence on regression."""
    X = np.random.randn(80, 10)
    y = np.random.randn(80)
    
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)
    
    result = partial_dependence(reg, X, features=[0, 1])
    
    assert len(result['values']) == 2
    assert len(result['predictions']) == 2


def test_partial_dependence_percentiles():
    """Test partial dependence with custom percentiles."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = partial_dependence(clf, X, features=[0], percentiles=(0.1, 0.9))
    
    assert len(result['values'][0]) == 100  # Default grid_resolution


def test_get_feature_importance_tree_based():
    """Test get_feature_importance with tree-based model."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = get_feature_importance(clf)
    
    assert 'importances' in result
    assert 'feature_names' in result
    assert 'sorted_idx' in result


def test_get_feature_importance_linear_model():
    """Test get_feature_importance with linear model."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    clf = LogisticRegression(max_iter=100)
    clf.fit(X, y)
    
    result = get_feature_importance(clf)
    
    assert len(result['importances']) == 10
    assert len(result['sorted_idx']) == 10


def test_get_feature_importance_feature_names():
    """Test get_feature_importance with custom feature names."""
    X = np.random.randn(80, 5)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    feature_names = ['age', 'income', 'score', 'balance', 'duration']
    result = get_feature_importance(clf, feature_names=feature_names)
    
    assert result['feature_names'] == feature_names


def test_get_feature_importance_sorted():
    """Test get_feature_importance sorting."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = get_feature_importance(clf)
    
    # Check that sorted_idx correctly orders by importance
    sorted_importances = result['importances'][result['sorted_idx']]
    assert all(sorted_importances[i] >= sorted_importances[i+1] 
               for i in range(len(sorted_importances)-1))


def test_get_feature_importance_error_no_attribute():
    """Test get_feature_importance error without required attributes."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    class DummyEstimator:
        def fit(self, X, y):
            return self
    
    estimator = DummyEstimator()
    estimator.fit(X, y)
    
    with pytest.raises(AttributeError):
        get_feature_importance(estimator)


def test_permutation_importance_positive_values():
    """Test permutation importance produces reasonable values."""
    # Create data where some features are informative
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Features 0,1 are informative
    
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    
    # Informative features should have higher importance
    assert result['importances_mean'][0] > 0
    assert result['importances_mean'][1] > 0


def test_partial_dependence_values_range():
    """Test partial dependence values are in reasonable range."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = partial_dependence(clf, X, features=[0])
    
    # For binary classification, predictions should be in [0, 1]
    assert all(0 <= p <= 1 for p in result['predictions'][0])


def test_permutation_importance_std():
    """Test permutation importance standard deviation."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
    
    # Standard deviation should be non-negative
    assert all(std >= 0 for std in result['importances_std'])


def test_get_feature_importance_default_names():
    """Test get_feature_importance generates default names."""
    X = np.random.randn(80, 7)
    y = np.random.randint(0, 2, 80)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    result = get_feature_importance(clf)
    
    # Should generate Feature 0, Feature 1, etc.
    assert result['feature_names'][0] == 'Feature 0'
    assert result['feature_names'][6] == 'Feature 6'
