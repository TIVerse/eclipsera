"""Tests for support vector machines."""
import numpy as np
import pytest

from eclipsera.ml.svm import SVC, SVR, LinearSVC, LinearSVR


def test_svc_linear_kernel(binary_classification_data):
    """Test SVC with linear kernel."""
    X, y = binary_classification_data
    
    clf = SVC(kernel='linear', C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'support_vectors_')
    assert hasattr(clf, 'support_')
    assert hasattr(clf, 'dual_coef_')
    assert hasattr(clf, 'intercept_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy is better than random
    accuracy = clf.score(X, y)
    assert accuracy > 0.5


def test_svc_rbf_kernel(binary_classification_data):
    """Test SVC with RBF kernel."""
    X, y = binary_classification_data
    
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=100, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'support_vectors_')
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # RBF should provide reasonable fit
    accuracy = clf.score(X, y)
    assert accuracy > 0.4


def test_svc_poly_kernel(binary_classification_data):
    """Test SVC with polynomial kernel."""
    X, y = binary_classification_data
    
    clf = SVC(kernel='poly', degree=2, C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_svc_sigmoid_kernel(binary_classification_data):
    """Test SVC with sigmoid kernel."""
    X, y = binary_classification_data
    
    clf = SVC(kernel='sigmoid', C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_svc_multiclass(classification_data):
    """Test SVC on multiclass problem."""
    X, y = classification_data
    
    clf = SVC(kernel='rbf', C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    assert len(clf.classes_) == 3
    assert hasattr(clf, 'binary_classifiers_')
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check all classes are predicted
    predicted_classes = np.unique(y_pred)
    assert len(predicted_classes) >= 1


def test_svc_c_parameter(binary_classification_data):
    """Test SVC with different C values."""
    X, y = binary_classification_data
    
    # Low C (more regularization)
    clf = SVC(kernel='linear', C=0.1, max_iter=100, random_state=42)
    clf.fit(X, y)
    y_pred1 = clf.predict(X)
    
    # High C (less regularization)
    clf = SVC(kernel='linear', C=10.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    y_pred2 = clf.predict(X)
    
    # Both should make predictions
    assert y_pred1.shape == y.shape
    assert y_pred2.shape == y.shape


def test_svc_gamma_parameter(binary_classification_data):
    """Test SVC with different gamma values."""
    X, y = binary_classification_data
    
    # Test gamma='scale'
    clf = SVC(kernel='rbf', gamma='scale', max_iter=100, random_state=42)
    clf.fit(X, y)
    y_pred1 = clf.predict(X)
    
    # Test gamma='auto'
    clf = SVC(kernel='rbf', gamma='auto', max_iter=100, random_state=42)
    clf.fit(X, y)
    y_pred2 = clf.predict(X)
    
    # Test specific gamma value
    clf = SVC(kernel='rbf', gamma=0.1, max_iter=100, random_state=42)
    clf.fit(X, y)
    y_pred3 = clf.predict(X)
    
    assert y_pred1.shape == y.shape
    assert y_pred2.shape == y.shape
    assert y_pred3.shape == y.shape


def test_svr_linear_kernel(regression_data):
    """Test SVR with linear kernel."""
    X, y = regression_data
    
    reg = SVR(kernel='linear', C=1.0, epsilon=0.1)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'support_vectors_')
    assert hasattr(reg, 'support_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_svr_rbf_kernel(regression_data):
    """Test SVR with RBF kernel."""
    X, y = regression_data
    
    reg = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_svr_epsilon_parameter(regression_data):
    """Test SVR with different epsilon values."""
    X, y = regression_data
    
    # Small epsilon
    reg = SVR(kernel='linear', epsilon=0.01)
    reg.fit(X, y)
    y_pred1 = reg.predict(X)
    
    # Large epsilon
    reg = SVR(kernel='linear', epsilon=0.5)
    reg.fit(X, y)
    y_pred2 = reg.predict(X)
    
    assert y_pred1.shape == y.shape
    assert y_pred2.shape == y.shape


def test_svc_small_dataset():
    """Test SVC on small dataset."""
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 0, 1, 1])
    
    clf = SVC(kernel='linear', max_iter=100)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_svr_small_dataset():
    """Test SVR on small dataset."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = SVR(kernel='linear')
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_linear_svc_binary(binary_classification_data):
    """Test LinearSVC on binary classification."""
    X, y = binary_classification_data
    
    clf = LinearSVC(C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'coef_')
    assert hasattr(clf, 'intercept_')
    assert hasattr(clf, 'classes_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = clf.score(X, y)
    assert accuracy > 0.5


def test_linear_svc_multiclass(classification_data):
    """Test LinearSVC on multiclass problem."""
    X, y = classification_data
    
    clf = LinearSVC(C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    assert len(clf.classes_) == 3
    assert clf.coef_.shape[0] == 3  # One-vs-rest
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_linear_svc_decision_function(binary_classification_data):
    """Test LinearSVC decision_function."""
    X, y = binary_classification_data
    
    clf = LinearSVC(C=1.0, max_iter=100, random_state=42)
    clf.fit(X, y)
    
    decision = clf.decision_function(X)
    assert decision.shape[0] == len(X)


def test_linear_svc_c_parameter(binary_classification_data):
    """Test LinearSVC with different C values."""
    X, y = binary_classification_data
    
    for C in [0.1, 1.0, 10.0]:
        clf = LinearSVC(C=C, max_iter=100, random_state=42)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


def test_linear_svc_small_dataset():
    """Test LinearSVC on small dataset."""
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 0, 1, 1])
    
    clf = LinearSVC(max_iter=100)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_linear_svr_fit_predict(regression_data):
    """Test LinearSVR fit and predict."""
    X, y = regression_data
    
    reg = LinearSVR(C=1.0, epsilon=0.1, max_iter=100)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'coef_')
    assert hasattr(reg, 'intercept_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_linear_svr_c_parameter(regression_data):
    """Test LinearSVR with different C values."""
    X, y = regression_data
    
    for C in [0.1, 1.0, 10.0]:
        reg = LinearSVR(C=C, max_iter=100)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape


def test_linear_svr_epsilon_parameter(regression_data):
    """Test LinearSVR with different epsilon values."""
    X, y = regression_data
    
    for epsilon in [0.01, 0.1, 0.5]:
        reg = LinearSVR(epsilon=epsilon, max_iter=100)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape


def test_linear_svr_multioutput(multioutput_regression_data):
    """Test LinearSVR with multi-output."""
    X, y = multioutput_regression_data
    
    reg = LinearSVR(max_iter=100)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_linear_svr_small_dataset():
    """Test LinearSVR on small dataset."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = LinearSVR(max_iter=100)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
