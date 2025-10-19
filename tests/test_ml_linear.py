"""Tests for linear models."""
import numpy as np
import pytest

from eclipsera.ml.linear import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)


def test_linear_regression_fit_predict(regression_data):
    """Test LinearRegression fit and predict."""
    X, y = regression_data
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert model.coef_.shape == (X.shape[1],)
    
    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score is reasonable
    score = model.score(X, y)
    assert score > 0.9  # Should fit well on training data


def test_linear_regression_no_intercept(regression_data):
    """Test LinearRegression without intercept."""
    X, y = regression_data
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    assert model.intercept_ == 0.0


def test_ridge_regression(regression_data):
    """Test Ridge regression."""
    X, y = regression_data
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Ridge should provide reasonable fit
    score = model.score(X, y)
    assert score > 0.8


def test_lasso_regression(regression_data):
    """Test Lasso regression."""
    X, y = regression_data
    
    model = Lasso(alpha=0.1, max_iter=1000)
    model.fit(X, y)
    
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Lasso should provide reasonable fit
    score = model.score(X, y)
    assert score > 0.7


def test_elastic_net(regression_data):
    """Test ElasticNet regression."""
    X, y = regression_data
    
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
    model.fit(X, y)
    
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_logistic_regression_binary(binary_classification_data):
    """Test LogisticRegression on binary classification."""
    X, y = binary_classification_data
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 2
    
    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert set(y_pred).issubset(set(y))
    
    # Check predict_proba
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    # Check accuracy is reasonable
    accuracy = model.score(X, y)
    assert accuracy > 0.5  # Better than random


def test_logistic_regression_multiclass(classification_data):
    """Test LogisticRegression on multiclass classification."""
    X, y = classification_data
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 3
    
    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Check predict_proba
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (len(X), 3)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_linear_regression_multioutput(multioutput_regression_data):
    """Test LinearRegression with multiple outputs."""
    X, y = multioutput_regression_data
    
    model = LinearRegression()
    model.fit(X, y)
    
    assert model.coef_.shape == (y.shape[1], X.shape[1])
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
