"""Tests for K-Nearest Neighbors algorithms."""
import numpy as np
import pytest

from eclipsera.ml.neighbors import KNeighborsClassifier, KNeighborsRegressor


def test_kneighbors_classifier_fit_predict(classification_data):
    """Test KNeighborsClassifier fit and predict."""
    X, y = classification_data
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')
    assert hasattr(clf, 'classes_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = clf.score(X, y)
    assert accuracy > 0.3


def test_kneighbors_classifier_predict_proba(binary_classification_data):
    """Test KNeighborsClassifier predict_proba."""
    X, y = binary_classification_data
    
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_kneighbors_classifier_n_neighbors():
    """Test KNeighborsClassifier with different n_neighbors."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    for n_neighbors in [1, 3, 5]:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


def test_kneighbors_classifier_uniform_weights(binary_classification_data):
    """Test KNeighborsClassifier with uniform weights."""
    X, y = binary_classification_data
    
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_distance_weights(binary_classification_data):
    """Test KNeighborsClassifier with distance weights."""
    X, y = binary_classification_data
    
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_euclidean_metric():
    """Test KNeighborsClassifier with Euclidean metric."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_manhattan_metric():
    """Test KNeighborsClassifier with Manhattan metric."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = KNeighborsClassifier(n_neighbors=2, metric='manhattan')
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_minkowski_metric():
    """Test KNeighborsClassifier with Minkowski metric."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=3)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_multiclass(classification_data):
    """Test KNeighborsClassifier on multiclass problem."""
    X, y = classification_data
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    
    assert len(clf.classes_) == 3
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)


def test_kneighbors_regressor_fit_predict(regression_data):
    """Test KNeighborsRegressor fit and predict."""
    X, y = regression_data
    
    reg = KNeighborsRegressor(n_neighbors=5)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'X_')
    assert hasattr(reg, 'y_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score is reasonable
    score = reg.score(X, y)
    assert score > 0.3


def test_kneighbors_regressor_n_neighbors():
    """Test KNeighborsRegressor with different n_neighbors."""
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    
    for n_neighbors in [1, 2, 3]:
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape


def test_kneighbors_regressor_uniform_weights(regression_data):
    """Test KNeighborsRegressor with uniform weights."""
    X, y = regression_data
    
    reg = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_regressor_distance_weights(regression_data):
    """Test KNeighborsRegressor with distance weights."""
    X, y = regression_data
    
    reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_regressor_euclidean_metric():
    """Test KNeighborsRegressor with Euclidean metric."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = KNeighborsRegressor(n_neighbors=2, metric='euclidean')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_regressor_manhattan_metric():
    """Test KNeighborsRegressor with Manhattan metric."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = KNeighborsRegressor(n_neighbors=2, metric='manhattan')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_regressor_multioutput(multioutput_regression_data):
    """Test KNeighborsRegressor with multi-output."""
    X, y = multioutput_regression_data
    
    reg = KNeighborsRegressor(n_neighbors=5)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_small_dataset():
    """Test KNeighborsClassifier on small dataset."""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 0])
    
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_regressor_small_dataset():
    """Test KNeighborsRegressor on small dataset."""
    X = np.array([[0], [1], [2]])
    y = np.array([0.0, 1.0, 2.0])
    
    reg = KNeighborsRegressor(n_neighbors=2)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_classifier_single_neighbor():
    """Test KNeighborsClassifier with n_neighbors=1."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    
    # With n_neighbors=1, training predictions should be perfect
    y_pred = clf.predict(X)
    assert np.all(y_pred == y)


def test_kneighbors_regressor_single_neighbor():
    """Test KNeighborsRegressor with n_neighbors=1."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = KNeighborsRegressor(n_neighbors=1)
    reg.fit(X, y)
    
    # With n_neighbors=1, training predictions should be exact
    y_pred = reg.predict(X)
    assert np.allclose(y_pred, y)


def test_kneighbors_classifier_all_neighbors():
    """Test KNeighborsClassifier with n_neighbors equal to dataset size."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = KNeighborsClassifier(n_neighbors=4, weights='uniform')
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_kneighbors_distance_computation():
    """Test that distances are computed correctly."""
    X_train = np.array([[0, 0], [3, 4]])  # Distance between points is 5
    y_train = np.array([0, 1])
    
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
    clf.fit(X_train, y_train)
    
    # Test point closer to [0, 0]
    X_test = np.array([[0.5, 0.5]])
    y_pred = clf.predict(X_test)
    assert y_pred[0] == 0
    
    # Test point closer to [3, 4]
    X_test = np.array([[2.5, 3.5]])
    y_pred = clf.predict(X_test)
    assert y_pred[0] == 1
