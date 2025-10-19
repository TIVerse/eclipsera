"""Tests for Recursive Feature Elimination."""
import numpy as np
import pytest

from eclipsera.feature_selection import RFE
from eclipsera.ml import LogisticRegression, RandomForestClassifier


def test_rfe_basic():
    """Test basic RFE functionality."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    assert hasattr(rfe, 'support_')
    assert hasattr(rfe, 'ranking_')
    assert hasattr(rfe, 'n_features_')
    assert rfe.n_features_ == 10


def test_rfe_transform():
    """Test RFE transform method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    X_transformed = rfe.fit_transform(X, y)
    
    assert X_transformed.shape == (50, 10)


def test_rfe_fit_transform():
    """Test RFE fit_transform method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=5)
    X_transformed = rfe.fit_transform(X, y)
    
    assert X_transformed.shape == (50, 5)
    assert hasattr(rfe, 'ranking_')


def test_rfe_get_support():
    """Test RFE get_support method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    # Boolean mask
    mask = rfe.get_support()
    assert mask.dtype == bool
    assert len(mask) == 20
    assert mask.sum() == 10
    
    # Indices
    indices = rfe.get_support(indices=True)
    assert len(indices) == 10


def test_rfe_predict():
    """Test RFE predict method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    y_pred = rfe.predict(X)
    
    assert y_pred.shape == (50,)


def test_rfe_score():
    """Test RFE score method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    score = rfe.score(X, y)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_rfe_ranking():
    """Test RFE produces valid ranking."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    # Selected features should have rank 1
    selected = rfe.support_
    assert np.all(rfe.ranking_[selected] == 1)
    
    # Rejected features should have rank > 1
    rejected = ~rfe.support_
    assert np.all(rfe.ranking_[rejected] > 1)


def test_rfe_n_features_to_select_none():
    """Test RFE with n_features_to_select=None (select half)."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=None)
    rfe.fit(X, y)
    
    # Should select half
    assert rfe.n_features_ == 10


def test_rfe_n_features_to_select_float():
    """Test RFE with float n_features_to_select (fraction)."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=0.5)
    rfe.fit(X, y)
    
    # Should select 50%
    assert rfe.n_features_ == 10


def test_rfe_step_parameter():
    """Test RFE with different step sizes."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    for step in [1, 3, 5]:
        estimator = LogisticRegression(max_iter=100)
        rfe = RFE(estimator, n_features_to_select=10, step=step)
        rfe.fit(X, y)
        
        assert rfe.n_features_ == 10


def test_rfe_step_float():
    """Test RFE with float step (percentage)."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=5, step=0.2)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 5


def test_rfe_with_random_forest():
    """Test RFE with RandomForestClassifier."""
    X = np.random.randn(50, 15)
    y = np.random.randint(0, 2, 50)
    
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    rfe = RFE(estimator, n_features_to_select=8)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 8


def test_rfe_selects_informative_features():
    """Test RFE selects informative features."""
    # Create data where first 5 features are informative
    X_informative = np.random.randn(100, 5)
    y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)
    
    # Add 10 random features
    X_noise = np.random.randn(100, 10)
    X = np.hstack([X_informative, X_noise])
    
    estimator = LogisticRegression(max_iter=200)
    rfe = RFE(estimator, n_features_to_select=5)
    rfe.fit(X, y)
    
    selected = rfe.get_support(indices=True)
    
    # At least some selected should be from first 5
    assert np.any(selected < 5)


def test_rfe_error_before_fit():
    """Test RFE raises error when calling methods before fit."""
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    X = np.random.randn(10, 20)
    y = np.random.randint(0, 2, 10)
    
    with pytest.raises(ValueError):
        rfe.transform(X)
    
    with pytest.raises(ValueError):
        rfe.get_support()
    
    with pytest.raises(ValueError):
        rfe.predict(X)
    
    with pytest.raises(ValueError):
        rfe.score(X, y)


def test_rfe_multiclass():
    """Test RFE on multiclass problem."""
    X = np.random.randn(60, 15)
    y = np.random.randint(0, 3, 60)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=8)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 8


def test_rfe_different_n_features():
    """Test RFE with different numbers of features to select."""
    X = np.random.randn(50, 30)
    y = np.random.randint(0, 2, 50)
    
    for n_features in [5, 10, 20]:
        estimator = LogisticRegression(max_iter=100)
        rfe = RFE(estimator, n_features_to_select=n_features)
        X_transformed = rfe.fit_transform(X, y)
        
        assert X_transformed.shape == (50, n_features)


def test_rfe_verbose_mode():
    """Test RFE with verbose mode."""
    X = np.random.randn(30, 15)
    y = np.random.randint(0, 2, 30)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=5, verbose=1)
    
    # Should not raise error with verbose
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 5


def test_rfe_estimator_fitted():
    """Test RFE stores fitted estimator."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    assert hasattr(rfe, 'estimator_')


def test_rfe_n_features_in():
    """Test RFE sets n_features_in_ attribute."""
    X = np.random.randn(50, 25)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=15)
    rfe.fit(X, y)
    
    assert rfe.n_features_in_ == 25


def test_rfe_in_pipeline():
    """Test RFE works in pipeline."""
    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    
    X = np.random.randn(60, 20)
    y = np.random.randint(0, 2, 60)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rfe', RFE(LogisticRegression(max_iter=100), n_features_to_select=10))
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == (60,)


def test_rfe_select_all_features():
    """Test RFE when selecting all features."""
    X = np.random.randn(50, 15)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=15)
    rfe.fit(X, y)
    
    # All features selected
    assert rfe.n_features_ == 15
    assert np.all(rfe.support_)


def test_rfe_select_one_feature():
    """Test RFE selecting only one feature."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=1)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 1
    assert rfe.support_.sum() == 1


def test_rfe_ranking_order():
    """Test RFE ranking is in correct order."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    
    # Rankings should be positive integers
    assert np.all(rfe.ranking_ >= 1)
    
    # Best features have rank 1
    assert np.sum(rfe.ranking_ == 1) == 10


def test_rfe_small_step():
    """Test RFE with step=1 (removes one feature at a time)."""
    X = np.random.randn(50, 15)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10, step=1)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 10


def test_rfe_reproducibility():
    """Test RFE is reproducible."""
    X = np.random.RandomState(42).randn(50, 20)
    y = np.random.RandomState(42).randint(0, 2, 50)
    
    rfe1 = RFE(RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=10)
    support1 = rfe1.fit(X, y).support_
    
    rfe2 = RFE(RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=10)
    support2 = rfe2.fit(X, y).support_
    
    # Should give same results
    assert np.array_equal(support1, support2)


def test_rfe_with_large_step():
    """Test RFE with large step size."""
    X = np.random.randn(50, 30)
    y = np.random.randint(0, 2, 50)
    
    estimator = LogisticRegression(max_iter=100)
    rfe = RFE(estimator, n_features_to_select=10, step=10)
    rfe.fit(X, y)
    
    assert rfe.n_features_ == 10
