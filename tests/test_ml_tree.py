"""Tests for decision tree algorithms."""
import numpy as np
import pytest

from eclipsera.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_decision_tree_classifier_fit_predict(classification_data):
    """Test DecisionTreeClassifier fit and predict."""
    X, y = classification_data
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'tree_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'feature_importances_')
    assert clf.n_classes_ == 3
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy is reasonable
    accuracy = clf.score(X, y)
    assert accuracy > 0.3  # Should do better than random


def test_decision_tree_classifier_max_depth(binary_classification_data):
    """Test DecisionTreeClassifier with max_depth."""
    X, y = binary_classification_data
    
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'tree_')
    
    # Shallow tree should still make predictions
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) > 0


def test_decision_tree_classifier_predict_proba(binary_classification_data):
    """Test DecisionTreeClassifier predict_proba."""
    X, y = binary_classification_data
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_decision_tree_classifier_entropy(classification_data):
    """Test DecisionTreeClassifier with entropy criterion."""
    X, y = classification_data
    
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.3


def test_decision_tree_classifier_feature_importances(binary_classification_data):
    """Test DecisionTreeClassifier feature importances."""
    X, y = binary_classification_data
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'feature_importances_')
    assert len(clf.feature_importances_) == X.shape[1]
    assert np.allclose(clf.feature_importances_.sum(), 1.0)


def test_decision_tree_regressor_fit_predict(regression_data):
    """Test DecisionTreeRegressor fit and predict."""
    X, y = regression_data
    
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'tree_')
    assert hasattr(reg, 'feature_importances_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score is reasonable
    score = reg.score(X, y)
    assert score > 0.5  # Should fit training data reasonably well


def test_decision_tree_regressor_max_depth(regression_data):
    """Test DecisionTreeRegressor with max_depth."""
    X, y = regression_data
    
    reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    
    # Shallow tree should still make reasonable predictions
    from eclipsera.core import mean_squared_error
    mse = mean_squared_error(y, y_pred)
    assert mse < np.var(y) * 2  # Should be better than predicting mean


def test_decision_tree_regressor_mae_criterion(regression_data):
    """Test DecisionTreeRegressor with MAE criterion."""
    X, y = regression_data
    
    reg = DecisionTreeRegressor(criterion='mae', max_depth=5, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    score = reg.score(X, y)
    assert score > 0.3


def test_decision_tree_min_samples_split(classification_data):
    """Test min_samples_split parameter."""
    X, y = classification_data
    
    clf = DecisionTreeClassifier(min_samples_split=20, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_decision_tree_min_samples_leaf(classification_data):
    """Test min_samples_leaf parameter."""
    X, y = classification_data
    
    clf = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_decision_tree_max_features(classification_data):
    """Test max_features parameter."""
    X, y = classification_data
    
    # Test with integer
    clf = DecisionTreeClassifier(max_features=2, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Test with float
    clf = DecisionTreeClassifier(max_features=0.5, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
