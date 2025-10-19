"""Tests for ensemble methods."""
import numpy as np
import pytest

from eclipsera.ml.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_random_forest_classifier_fit_predict(classification_data):
    """Test RandomForestClassifier fit and predict."""
    X, y = classification_data
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'estimators_')
    assert len(clf.estimators_) == 10
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'feature_importances_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = clf.score(X, y)
    assert accuracy > 0.3


def test_random_forest_classifier_predict_proba(binary_classification_data):
    """Test RandomForestClassifier predict_proba."""
    X, y = binary_classification_data
    
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_random_forest_classifier_feature_importances(binary_classification_data):
    """Test RandomForestClassifier feature importances."""
    X, y = binary_classification_data
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'feature_importances_')
    assert len(clf.feature_importances_) == X.shape[1]
    assert np.allclose(clf.feature_importances_.sum(), 1.0)


def test_random_forest_classifier_no_bootstrap(classification_data):
    """Test RandomForestClassifier without bootstrap."""
    X, y = classification_data
    
    clf = RandomForestClassifier(
        n_estimators=5,
        bootstrap=False,
        random_state=42
    )
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_random_forest_regressor_fit_predict(regression_data):
    """Test RandomForestRegressor fit and predict."""
    X, y = regression_data
    
    reg = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'estimators_')
    assert len(reg.estimators_) == 10
    assert hasattr(reg, 'feature_importances_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score
    score = reg.score(X, y)
    assert score > 0.5


def test_random_forest_regressor_feature_importances(regression_data):
    """Test RandomForestRegressor feature importances."""
    X, y = regression_data
    
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)
    
    assert hasattr(reg, 'feature_importances_')
    assert len(reg.feature_importances_) == X.shape[1]


def test_random_forest_max_features(classification_data):
    """Test RandomForest with different max_features."""
    X, y = classification_data
    
    # Test with 'sqrt'
    clf = RandomForestClassifier(n_estimators=5, max_features='sqrt', random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Test with 'log2'
    clf = RandomForestClassifier(n_estimators=5, max_features='log2', random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_gradient_boosting_classifier_fit_predict(binary_classification_data):
    """Test GradientBoostingClassifier fit and predict."""
    X, y = binary_classification_data
    
    clf = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'estimators_')
    assert len(clf.estimators_) == 10
    assert hasattr(clf, 'classes_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = clf.score(X, y)
    assert accuracy > 0.5


def test_gradient_boosting_classifier_predict_proba(binary_classification_data):
    """Test GradientBoostingClassifier predict_proba."""
    X, y = binary_classification_data
    
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_gradient_boosting_classifier_multiclass(classification_data):
    """Test GradientBoostingClassifier on multiclass problem."""
    X, y = classification_data
    
    clf = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    )
    clf.fit(X, y)
    
    assert len(clf.classes_) == 3
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)


def test_gradient_boosting_classifier_subsample(binary_classification_data):
    """Test GradientBoostingClassifier with subsampling."""
    X, y = binary_classification_data
    
    clf = GradientBoostingClassifier(
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_gradient_boosting_regressor_fit_predict(regression_data):
    """Test GradientBoostingRegressor fit and predict."""
    X, y = regression_data
    
    reg = GradientBoostingRegressor(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'estimators_')
    assert len(reg.estimators_) == 10
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score
    score = reg.score(X, y)
    assert score > 0.5


def test_gradient_boosting_regressor_learning_rate(regression_data):
    """Test GradientBoostingRegressor with different learning rates."""
    X, y = regression_data
    
    # Low learning rate
    reg = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.01,
        max_depth=2,
        random_state=42
    )
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    score1 = reg.score(X, y)
    
    # Higher learning rate
    reg = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.5,
        max_depth=2,
        random_state=42
    )
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    score2 = reg.score(X, y)
    
    # Both should produce reasonable fits (at least better than always predicting mean)
    assert score1 > 0.0  # Should be positive R²
    assert score2 > 0.0


def test_gradient_boosting_regressor_subsample(regression_data):
    """Test GradientBoostingRegressor with subsampling."""
    X, y = regression_data
    
    reg = GradientBoostingRegressor(
        n_estimators=10,
        subsample=0.7,
        random_state=42
    )
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
