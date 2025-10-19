"""Tests for model selection splitting and cross-validation."""
import numpy as np
import pytest

from eclipsera.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from eclipsera.ml import LogisticRegression


def test_train_test_split_basic():
    """Test basic train_test_split functionality."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 2


def test_train_test_split_no_shuffle():
    """Test train_test_split without shuffling."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # Without shuffle, should be sequential split
    assert np.array_equal(y_test, np.array([0, 1]))


def test_train_test_split_stratify():
    """Test train_test_split with stratification."""
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 25 + [1] * 25)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Check class distribution is preserved
    train_ratio = np.sum(y_train == 0) / len(y_train)
    test_ratio = np.sum(y_test == 0) / len(y_test)
    
    assert abs(train_ratio - 0.5) < 0.1
    assert abs(test_ratio - 0.5) < 0.1


def test_train_test_split_train_size():
    """Test train_test_split with train_size."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)
    
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2


def test_train_test_split_multiple_arrays():
    """Test train_test_split with multiple arrays."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    z = np.array([1, 2, 3, 4, 5])
    
    result = train_test_split(X, y, z, test_size=0.4, random_state=42)
    
    assert len(result) == 6  # 3 arrays × 2 (train/test)
    X_train, X_test, y_train, y_test, z_train, z_test = result
    
    assert X_train.shape[0] == y_train.shape[0] == z_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0] == z_test.shape[0]


def test_train_test_split_errors():
    """Test train_test_split error handling."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0])  # Wrong size
    
    with pytest.raises(ValueError):
        train_test_split(X, y)
    
    # Test size out of range
    with pytest.raises(ValueError):
        train_test_split(X, np.array([0, 1, 0, 1, 0]), test_size=1.5)


def test_kfold_basic():
    """Test basic KFold functionality."""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    
    kf = KFold(n_splits=5)
    
    splits = list(kf.split(X, y))
    assert len(splits) == 5
    
    for train_idx, test_idx in splits:
        assert len(train_idx) == 4
        assert len(test_idx) == 1


def test_kfold_shuffle():
    """Test KFold with shuffling."""
    X = np.arange(20).reshape(10, 2)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X))
    
    # With shuffle, indices should not be sequential
    first_test = splits[0][1]
    assert not np.array_equal(first_test, np.array([0, 1]))


def test_kfold_get_n_splits():
    """Test KFold get_n_splits method."""
    kf = KFold(n_splits=3)
    assert kf.get_n_splits() == 3


def test_kfold_uneven_splits():
    """Test KFold with uneven number of samples."""
    X = np.arange(14).reshape(7, 2)
    
    kf = KFold(n_splits=3)
    splits = list(kf.split(X))
    
    assert len(splits) == 3
    
    # Check all samples are used
    all_test_indices = np.concatenate([test for _, test in splits])
    assert len(np.unique(all_test_indices)) == 7


def test_kfold_error_n_splits():
    """Test KFold error with invalid n_splits."""
    with pytest.raises(ValueError):
        KFold(n_splits=1)


def test_stratified_kfold_basic():
    """Test basic StratifiedKFold functionality."""
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    skf = StratifiedKFold(n_splits=5)
    
    splits = list(skf.split(X, y))
    assert len(splits) == 5
    
    # Check stratification
    for train_idx, test_idx in splits:
        y_test = y[test_idx]
        # Each fold should have balanced classes
        assert len(y_test) == 2
        assert len(np.unique(y_test)) == 2


def test_stratified_kfold_multiclass():
    """Test StratifiedKFold with multiclass."""
    X = np.arange(30).reshape(15, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    
    skf = StratifiedKFold(n_splits=5)
    
    splits = list(skf.split(X, y))
    
    # Check each fold has all classes
    for train_idx, test_idx in splits:
        y_test = y[test_idx]
        assert len(np.unique(y_test)) == 3


def test_stratified_kfold_shuffle():
    """Test StratifiedKFold with shuffling."""
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    
    assert len(splits) == 5


def test_stratified_kfold_error_too_few_samples():
    """Test StratifiedKFold error with too few samples per class."""
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 0, 1, 1])
    
    skf = StratifiedKFold(n_splits=5)
    
    with pytest.raises(ValueError, match="least populated class"):
        list(skf.split(X, y))


def test_cross_val_score_basic(binary_classification_data):
    """Test basic cross_val_score functionality."""
    X, y = binary_classification_data
    
    clf = LogisticRegression(max_iter=100)
    scores = cross_val_score(clf, X, y, cv=3)
    
    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores)


def test_cross_val_score_with_kfold(binary_classification_data):
    """Test cross_val_score with explicit KFold."""
    X, y = binary_classification_data
    
    clf = LogisticRegression(max_iter=100)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    
    assert len(scores) == 5


def test_cross_val_score_default_cv(binary_classification_data):
    """Test cross_val_score with default cv."""
    X, y = binary_classification_data
    
    clf = LogisticRegression(max_iter=100)
    scores = cross_val_score(clf, X, y)
    
    assert len(scores) == 5  # Default is 5-fold


def test_cross_validate_basic(binary_classification_data):
    """Test basic cross_validate functionality."""
    X, y = binary_classification_data
    
    clf = LogisticRegression(max_iter=100)
    cv_results = cross_validate(clf, X, y, cv=3)
    
    assert 'test_score' in cv_results
    assert len(cv_results['test_score']) == 3


def test_cross_validate_return_train_score(binary_classification_data):
    """Test cross_validate with return_train_score."""
    X, y = binary_classification_data
    
    clf = LogisticRegression(max_iter=100)
    cv_results = cross_validate(clf, X, y, cv=3, return_train_score=True)
    
    assert 'test_score' in cv_results
    assert 'train_score' in cv_results
    assert len(cv_results['train_score']) == 3


def test_cross_validate_with_stratified(classification_data):
    """Test cross_validate on multiclass data."""
    X, y = classification_data
    
    clf = LogisticRegression(max_iter=100)
    cv_results = cross_validate(clf, X, y, cv=3)
    
    assert len(cv_results['test_score']) == 3


def test_kfold_coverage():
    """Test KFold covers all samples exactly once per fold."""
    X = np.arange(20).reshape(10, 2)
    
    kf = KFold(n_splits=5)
    
    for train_idx, test_idx in kf.split(X):
        # No overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # All samples covered
        assert len(train_idx) + len(test_idx) == len(X)


def test_stratified_kfold_maintains_distribution():
    """Test that StratifiedKFold maintains class distribution."""
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 30 + [1] * 20)  # 60-40 split
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in skf.split(X, y):
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Check proportions are maintained (with some tolerance)
        train_ratio = np.sum(y_train == 0) / len(y_train)
        test_ratio = np.sum(y_test == 0) / len(y_test)
        
        assert abs(train_ratio - 0.6) < 0.1
        assert abs(test_ratio - 0.6) < 0.1
