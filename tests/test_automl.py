"""Tests for AutoML functionality."""
import numpy as np
import pytest

from eclipsera.automl import AutoClassifier, AutoRegressor


def test_autoclassifier_basic():
    """Test basic AutoClassifier functionality."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    assert hasattr(auto_clf, 'best_estimator_')
    assert hasattr(auto_clf, 'best_algorithm_')
    assert hasattr(auto_clf, 'best_score_')
    assert hasattr(auto_clf, 'scores_')


def test_autoclassifier_predict():
    """Test AutoClassifier predict method."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    y_pred = auto_clf.predict(X)
    
    assert y_pred.shape == (100,)
    assert all(label in [0, 1] for label in y_pred)


def test_autoclassifier_predict_proba():
    """Test AutoClassifier predict_proba method."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    if hasattr(auto_clf.best_estimator_, 'predict_proba'):
        proba = auto_clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_autoclassifier_score():
    """Test AutoClassifier score method."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    score = auto_clf.score(X, y)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_autoclassifier_multiclass():
    """Test AutoClassifier on multiclass problem."""
    X = np.random.randn(120, 10)
    y = np.random.randint(0, 3, 120)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    y_pred = auto_clf.predict(X)
    
    assert len(np.unique(y_pred)) <= 3


def test_autoclassifier_algorithm_selection():
    """Test AutoClassifier selects specific algorithms."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    auto_clf = AutoClassifier(
        algorithms=['logistic_regression', 'random_forest'],
        cv=3,
        verbose=0,
        random_state=42
    )
    auto_clf.fit(X, y)
    
    assert auto_clf.best_algorithm_ in ['logistic_regression', 'random_forest']
    assert len(auto_clf.scores_) == 2


def test_autoclassifier_error_before_fit():
    """Test AutoClassifier raises error before fit."""
    auto_clf = AutoClassifier()
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        auto_clf.predict(X)


def test_autoclassifier_verbose():
    """Test AutoClassifier verbose mode."""
    X = np.random.randn(60, 8)
    y = np.random.randint(0, 2, 60)
    
    auto_clf = AutoClassifier(cv=2, verbose=1, random_state=42)
    auto_clf.fit(X, y)
    
    # Should not raise error with verbose
    assert auto_clf.best_score_ is not None


def test_autoclassifier_classes():
    """Test AutoClassifier stores classes."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    assert hasattr(auto_clf, 'classes_')
    assert len(auto_clf.classes_) == 2


def test_autoregressor_basic():
    """Test basic AutoRegressor functionality."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    auto_reg = AutoRegressor(cv=3, verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    assert hasattr(auto_reg, 'best_estimator_')
    assert hasattr(auto_reg, 'best_algorithm_')
    assert hasattr(auto_reg, 'best_score_')
    assert hasattr(auto_reg, 'scores_')


def test_autoregressor_predict():
    """Test AutoRegressor predict method."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    auto_reg = AutoRegressor(cv=3, verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    y_pred = auto_reg.predict(X)
    
    assert y_pred.shape == (100,)
    assert all(isinstance(val, (float, np.floating)) for val in y_pred)


def test_autoregressor_score():
    """Test AutoRegressor score method."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    auto_reg = AutoRegressor(cv=3, scoring='r2', verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    score = auto_reg.score(X, y)
    
    assert isinstance(score, float)


def test_autoregressor_algorithm_selection():
    """Test AutoRegressor selects specific algorithms."""
    X = np.random.randn(80, 10)
    y = np.random.randn(80)
    
    auto_reg = AutoRegressor(
        algorithms=['linear_regression', 'ridge'],
        cv=3,
        verbose=0,
        random_state=42
    )
    auto_reg.fit(X, y)
    
    assert auto_reg.best_algorithm_ in ['linear_regression', 'ridge']
    assert len(auto_reg.scores_) == 2


def test_autoregressor_scoring():
    """Test AutoRegressor with different scoring methods."""
    X = np.random.randn(80, 10)
    y = np.random.randn(80)
    
    for scoring in ['r2', 'neg_mean_squared_error']:
        auto_reg = AutoRegressor(cv=3, scoring=scoring, verbose=0, random_state=42)
        auto_reg.fit(X, y)
        
        assert auto_reg.best_score_ is not None


def test_autoregressor_error_before_fit():
    """Test AutoRegressor raises error before fit."""
    auto_reg = AutoRegressor()
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        auto_reg.predict(X)


def test_autoclassifier_n_features_in():
    """Test AutoClassifier sets n_features_in_."""
    X = np.random.randn(80, 15)
    y = np.random.randint(0, 2, 80)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    assert auto_clf.n_features_in_ == 15


def test_autoregressor_n_features_in():
    """Test AutoRegressor sets n_features_in_."""
    X = np.random.randn(80, 12)
    y = np.random.randn(80)
    
    auto_reg = AutoRegressor(cv=3, verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    assert auto_reg.n_features_in_ == 12


def test_autoclassifier_best_algorithm_exists():
    """Test AutoClassifier best algorithm is from candidates."""
    X = np.random.randn(80, 10)
    y = np.random.randint(0, 2, 80)
    
    auto_clf = AutoClassifier(cv=3, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    valid_algorithms = [
        'logistic_regression', 'random_forest', 'gradient_boosting',
        'knn', 'decision_tree', 'naive_bayes'
    ]
    
    assert auto_clf.best_algorithm_ in valid_algorithms


def test_autoregressor_best_algorithm_exists():
    """Test AutoRegressor best algorithm is from candidates."""
    X = np.random.randn(80, 10)
    y = np.random.randn(80)
    
    auto_reg = AutoRegressor(cv=3, verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    valid_algorithms = [
        'linear_regression', 'ridge', 'lasso', 'random_forest',
        'gradient_boosting', 'knn', 'decision_tree'
    ]
    
    assert auto_reg.best_algorithm_ in valid_algorithms


def test_autoclassifier_small_dataset():
    """Test AutoClassifier on small dataset."""
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, 30)
    
    auto_clf = AutoClassifier(cv=2, verbose=0, random_state=42)
    auto_clf.fit(X, y)
    
    y_pred = auto_clf.predict(X)
    assert y_pred.shape == (30,)


def test_autoregressor_small_dataset():
    """Test AutoRegressor on small dataset."""
    X = np.random.randn(30, 5)
    y = np.random.randn(30)
    
    auto_reg = AutoRegressor(cv=2, verbose=0, random_state=42)
    auto_reg.fit(X, y)
    
    y_pred = auto_reg.predict(X)
    assert y_pred.shape == (30,)
