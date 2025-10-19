"""Tests for feature selection."""
import numpy as np
import pytest

from eclipsera.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
)


def test_variance_threshold_basic():
    """Test basic VarianceThreshold functionality."""
    X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)
    
    assert hasattr(selector, 'variances_')
    assert len(selector.variances_) == 4


def test_variance_threshold_transform():
    """Test VarianceThreshold transform method."""
    X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    
    selector = VarianceThreshold(threshold=0.1)
    X_transformed = selector.fit_transform(X)
    
    # First column has zero variance, should be removed
    assert X_transformed.shape[1] < 4


def test_variance_threshold_zero_variance():
    """Test VarianceThreshold removes zero variance features."""
    X = np.array([[0, 2, 1], [0, 3, 2], [0, 4, 3]])
    
    selector = VarianceThreshold(threshold=0.0)
    X_transformed = selector.fit_transform(X)
    
    # First column has zero variance
    assert X_transformed.shape == (3, 2)


def test_variance_threshold_all_high_variance():
    """Test VarianceThreshold when all features have high variance."""
    X = np.random.randn(20, 5)
    
    selector = VarianceThreshold(threshold=0.1)
    X_transformed = selector.fit_transform(X)
    
    # Should keep all features
    assert X_transformed.shape == X.shape


def test_variance_threshold_get_support():
    """Test VarianceThreshold get_support method."""
    X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)
    
    # Boolean mask
    mask = selector.get_support()
    assert mask.dtype == bool
    assert len(mask) == 4
    
    # Indices
    indices = selector.get_support(indices=True)
    assert all(isinstance(idx, (int, np.integer)) for idx in indices)


def test_variance_threshold_different_thresholds():
    """Test VarianceThreshold with different threshold values."""
    X = np.random.randn(50, 10)
    
    for threshold in [0.0, 0.5, 1.0]:
        selector = VarianceThreshold(threshold=threshold)
        X_transformed = selector.fit_transform(X)
        
        # Higher threshold should remove more features
        assert X_transformed.shape[1] <= 10


def test_variance_threshold_error_before_fit():
    """Test VarianceThreshold raises error before fit."""
    selector = VarianceThreshold()
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError):
        selector.transform(X)
    
    with pytest.raises(ValueError):
        selector.get_support()


def test_variance_threshold_error_no_features():
    """Test VarianceThreshold raises error when all features removed."""
    X = np.ones((10, 3))  # All constant
    
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)
    
    with pytest.raises(ValueError, match="No features"):
        selector.transform(X)


def test_f_classif_basic():
    """Test f_classif scoring function."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    F, pval = f_classif(X, y)
    
    assert F.shape == (10,)
    assert pval.shape == (10,)
    assert np.all(F >= 0)


def test_f_classif_multiclass():
    """Test f_classif with multiclass problem."""
    X = np.random.randn(60, 5)
    y = np.random.randint(0, 3, 60)
    
    F, pval = f_classif(X, y)
    
    assert F.shape == (5,)
    assert np.all(F >= 0)


def test_chi2_basic():
    """Test chi2 scoring function."""
    X = np.abs(np.random.randn(50, 10))  # Non-negative
    y = np.random.randint(0, 2, 50)
    
    chi2_scores, pval = chi2(X, y)
    
    assert chi2_scores.shape == (10,)
    assert pval.shape == (10,)
    assert np.all(chi2_scores >= 0)


def test_chi2_error_negative():
    """Test chi2 raises error for negative values."""
    X = np.random.randn(50, 10)  # Can have negative values
    y = np.random.randint(0, 2, 50)
    
    with pytest.raises(ValueError, match="non-negative"):
        chi2(X, y)


def test_select_kbest_basic():
    """Test basic SelectKBest functionality."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=10)
    selector.fit(X, y)
    
    assert hasattr(selector, 'scores_')
    assert hasattr(selector, 'pvalues_')
    assert len(selector.scores_) == 20


def test_select_kbest_transform():
    """Test SelectKBest transform method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=10)
    X_transformed = selector.fit_transform(X, y)
    
    assert X_transformed.shape == (50, 10)


def test_select_kbest_fit_transform():
    """Test SelectKBest fit_transform method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=5)
    X_transformed = selector.fit_transform(X, y)
    
    assert X_transformed.shape == (50, 5)
    assert hasattr(selector, 'scores_')


def test_select_kbest_get_support():
    """Test SelectKBest get_support method."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    
    # Boolean mask
    mask = selector.get_support()
    assert mask.dtype == bool
    assert len(mask) == 20
    assert mask.sum() == 5
    
    # Indices
    indices = selector.get_support(indices=True)
    assert len(indices) == 5


def test_select_kbest_different_k():
    """Test SelectKBest with different k values."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    for k in [5, 10, 15]:
        selector = SelectKBest(f_classif, k=k)
        X_transformed = selector.fit_transform(X, y)
        
        assert X_transformed.shape == (50, k)


def test_select_kbest_k_all():
    """Test SelectKBest with k='all'."""
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k='all')
    X_transformed = selector.fit_transform(X, y)
    
    # Should keep all features
    assert X_transformed.shape == X.shape


def test_select_kbest_k_larger_than_features():
    """Test SelectKBest when k > n_features."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=20)  # More than 10 features
    X_transformed = selector.fit_transform(X, y)
    
    # Should keep all 10 features
    assert X_transformed.shape == (50, 10)


def test_select_kbest_with_chi2():
    """Test SelectKBest with chi2 scoring function."""
    X = np.abs(np.random.randn(50, 20))
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(chi2, k=10)
    X_transformed = selector.fit_transform(X, y)
    
    assert X_transformed.shape == (50, 10)


def test_select_kbest_selects_best():
    """Test SelectKBest actually selects best features."""
    # Create data where first 5 features are informative
    X_informative = np.random.randn(100, 5)
    y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)
    
    # Add 10 random features
    X_noise = np.random.randn(100, 10)
    X = np.hstack([X_informative, X_noise])
    
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    
    # Best features should be mostly from first 5
    selected_indices = selector.get_support(indices=True)
    # At least some should be from first 5
    assert np.any(selected_indices < 5)


def test_select_kbest_error_before_fit():
    """Test SelectKBest raises error before fit."""
    selector = SelectKBest(f_classif, k=5)
    X = np.random.randn(10, 20)
    
    with pytest.raises(ValueError):
        selector.transform(X)
    
    with pytest.raises(ValueError):
        selector.get_support()


def test_select_kbest_multiclass():
    """Test SelectKBest on multiclass problem."""
    X = np.random.randn(60, 15)
    y = np.random.randint(0, 3, 60)
    
    selector = SelectKBest(f_classif, k=8)
    X_transformed = selector.fit_transform(X, y)
    
    assert X_transformed.shape == (60, 8)


def test_variance_threshold_in_pipeline():
    """Test VarianceThreshold in pipeline."""
    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    
    X = np.random.randn(50, 10)
    
    pipe = Pipeline([
        ('variance', VarianceThreshold(0.1)),
        ('scaler', StandardScaler())
    ])
    
    X_transformed = pipe.fit_transform(X)
    assert X_transformed.shape[0] == 50


def test_select_kbest_in_pipeline():
    """Test SelectKBest in pipeline."""
    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    from eclipsera.ml import LogisticRegression
    
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    
    pipe = Pipeline([
        ('selector', SelectKBest(f_classif, k=10)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_variance_threshold_reproducibility():
    """Test VarianceThreshold is reproducible."""
    X = np.random.RandomState(42).randn(50, 10)
    
    selector1 = VarianceThreshold(threshold=0.5)
    X1 = selector1.fit_transform(X)
    
    selector2 = VarianceThreshold(threshold=0.5)
    X2 = selector2.fit_transform(X)
    
    assert np.array_equal(X1, X2)


def test_select_kbest_scores_ordering():
    """Test SelectKBest selects features by score order."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    
    # Get selected indices
    selected_indices = selector.get_support(indices=True)
    selected_scores = selector.scores_[selected_indices]
    
    # Selected scores should be among the top scores
    all_scores_sorted = np.sort(selector.scores_)[::-1]
    top_5_scores = all_scores_sorted[:5]
    
    # All selected scores should be in top 5
    for score in selected_scores:
        assert score in top_5_scores or np.isclose(score, top_5_scores).any()


def test_f_classif_two_classes():
    """Test f_classif with two clear classes."""
    # Create data where feature 0 separates classes well
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    
    F, pval = f_classif(X, y)
    
    # Feature 0 should have highest F-score
    assert F[0] == np.max(F)


def test_chi2_binary_features():
    """Test chi2 with binary features."""
    X = np.random.randint(0, 2, (100, 10)).astype(float)
    y = np.random.randint(0, 2, 100)
    
    chi2_scores, pval = chi2(X, y)
    
    assert chi2_scores.shape == (10,)
    assert np.all(chi2_scores >= 0)
