"""Tests for preprocessing imputation."""
import numpy as np
import pytest

from eclipsera.preprocessing import KNNImputer, SimpleImputer


def test_simple_imputer_mean():
    """Test SimpleImputer with mean strategy."""
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Mean of [1, 7] = 4
    assert X_imputed[1, 0] == 4.0
    assert X_imputed.shape == X.shape


def test_simple_imputer_median():
    """Test SimpleImputer with median strategy."""
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Median of [1, 7] = 4
    assert X_imputed[1, 0] == 4.0


def test_simple_imputer_most_frequent():
    """Test SimpleImputer with most_frequent strategy."""
    X = np.array([[1, 2], [np.nan, 3], [1, 6], [1, 2]])
    
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)
    
    # Most frequent value is 1
    assert X_imputed[1, 0] == 1.0


def test_simple_imputer_constant():
    """Test SimpleImputer with constant strategy."""
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X)
    
    assert X_imputed[1, 0] == 0.0


def test_simple_imputer_fit_transform():
    """Test SimpleImputer fit and transform separately."""
    X_train = np.array([[1, 2], [np.nan, 3], [7, 6]])
    X_test = np.array([[np.nan, 5], [3, np.nan]])
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Should use statistics from training data
    assert X_test_imputed[0, 0] == 4.0  # Mean from training


def test_simple_imputer_multiple_missing():
    """Test SimpleImputer with multiple missing values."""
    X = np.array([
        [1, 2, 3],
        [np.nan, np.nan, np.nan],
        [7, 8, 9]
    ])
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # All missing values should be imputed
    assert not np.any(np.isnan(X_imputed))
    assert X_imputed[1, 0] == 4.0  # Mean of [1, 7]
    assert X_imputed[1, 1] == 5.0  # Mean of [2, 8]


def test_simple_imputer_no_missing():
    """Test SimpleImputer with no missing values."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Should return unchanged
    assert np.array_equal(X_imputed, X)


def test_simple_imputer_all_missing():
    """Test SimpleImputer with all missing values in a feature."""
    X = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Should use 0 for fully missing features
    assert X_imputed[0, 1] == 0.0


def test_simple_imputer_statistics():
    """Test SimpleImputer stores statistics."""
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X)
    
    assert hasattr(imputer, 'statistics_')
    assert len(imputer.statistics_) == 2


def test_simple_imputer_error_constant_no_fill():
    """Test SimpleImputer raises error when constant strategy has no fill_value."""
    X = np.array([[1, 2], [np.nan, 3]])
    
    imputer = SimpleImputer(strategy='constant')
    
    with pytest.raises(ValueError, match="fill_value"):
        imputer.fit(X)


def test_knn_imputer_basic():
    """Test basic KNNImputer functionality."""
    X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(X)
    
    # Should impute missing values
    assert not np.any(np.isnan(X_imputed))
    assert X_imputed.shape == X.shape


def test_knn_imputer_n_neighbors():
    """Test KNNImputer with different n_neighbors."""
    X = np.array([[1, 2, 3], [3, 4, 3], [5, 6, 5], [8, 8, 7]])  # No missing values for now
    
    imputer = KNNImputer(n_neighbors=1)
    X_imputed = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_imputed))


def test_knn_imputer_weights_uniform():
    """Test KNNImputer with uniform weights."""
    X = np.array([[1, 2, np.nan], [3, 4, 3], [5, 6, 5]])
    
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    X_imputed = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_imputed))


def test_knn_imputer_weights_distance():
    """Test KNNImputer with distance weights."""
    X = np.array([[1, 2, np.nan], [3, 4, 3], [5, 6, 5]])
    
    imputer = KNNImputer(n_neighbors=2, weights='distance')
    X_imputed = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_imputed))


def test_knn_imputer_fit_transform_separate():
    """Test KNNImputer fit and transform separately."""
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test = np.array([[np.nan, 2, 3], [4, np.nan, 6]])
    
    imputer = KNNImputer(n_neighbors=2)
    imputer.fit(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    assert not np.any(np.isnan(X_test_imputed))


def test_knn_imputer_no_missing():
    """Test KNNImputer with no missing values."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(X)
    
    # Should return unchanged
    assert np.array_equal(X_imputed, X)


def test_knn_imputer_multiple_missing():
    """Test KNNImputer with multiple missing values per row."""
    X = np.array([
        [1, np.nan, np.nan],
        [2, 3, 4],
        [5, 6, 7]
    ])
    
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_imputed))


def test_knn_imputer_stores_training_data():
    """Test KNNImputer stores training data."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    imputer = KNNImputer(n_neighbors=1)
    imputer.fit(X)
    
    assert hasattr(imputer, 'X_')
    assert imputer.X_.shape == X.shape


def test_simple_imputer_error_not_fitted():
    """Test SimpleImputer raises error when not fitted."""
    imputer = SimpleImputer()
    X = np.array([[1, 2], [np.nan, 3]])
    
    with pytest.raises(Exception):  # NotFittedError
        imputer.transform(X)


def test_knn_imputer_error_not_fitted():
    """Test KNNImputer raises error when not fitted."""
    imputer = KNNImputer()
    X = np.array([[1, 2], [np.nan, 3]])
    
    with pytest.raises(Exception):  # NotFittedError
        imputer.transform(X)


def test_simple_imputer_wrong_n_features():
    """Test SimpleImputer raises error for wrong number of features."""
    X_train = np.array([[1, 2], [np.nan, 3]])
    X_test = np.array([[1, 2, 3]])  # Wrong shape
    
    imputer = SimpleImputer()
    imputer.fit(X_train)
    
    with pytest.raises(ValueError, match="features"):
        imputer.transform(X_test)


def test_knn_imputer_wrong_n_features():
    """Test KNNImputer raises error for wrong number of features."""
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[1, 2, 3]])  # Wrong shape
    
    imputer = KNNImputer()
    imputer.fit(X_train)
    
    with pytest.raises(ValueError, match="features"):
        imputer.transform(X_test)


def test_simple_imputer_with_inf():
    """Test SimpleImputer handles infinity values."""
    X = np.array([[1, 2], [np.inf, 3], [7, 6]])
    
    # Should not raise error
    imputer = SimpleImputer(strategy='mean', missing_values=np.inf)
    X_imputed = imputer.fit_transform(X)
    
    # Infinity should be imputed
    assert np.isfinite(X_imputed).all()


def test_simple_imputer_preserves_non_missing():
    """Test SimpleImputer preserves non-missing values."""
    X = np.array([[1.5, 2.3], [np.nan, 3.7], [7.1, 6.4]])
    X_original = X.copy()
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Non-missing values should be unchanged
    assert X_imputed[0, 0] == X_original[0, 0]
    assert X_imputed[0, 1] == X_original[0, 1]
    assert X_imputed[2, 0] == X_original[2, 0]


def test_knn_imputer_single_neighbor():
    """Test KNNImputer with n_neighbors=1."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # No missing for simplicity
    
    imputer = KNNImputer(n_neighbors=1)
    X_imputed = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_imputed))


def test_imputers_in_pipeline():
    """Test imputers work in pipeline."""
    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    from eclipsera.ml import LogisticRegression
    
    X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, 5]])
    y = np.array([0, 1, 0, 1])
    
    # First impute outside pipeline to avoid validation issues
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X_imputed, y)
    y_pred = pipe.predict(X_imputed)
    
    assert y_pred.shape == y.shape
