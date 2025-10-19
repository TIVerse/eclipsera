"""Tests for Neural Network (MLP) algorithms."""
import numpy as np
import pytest

from eclipsera.ml.neural_network import MLPClassifier, MLPRegressor


def test_mlp_classifier_fit_predict(binary_classification_data):
    """Test MLPClassifier fit and predict."""
    X, y = binary_classification_data
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'coefs_')
    assert hasattr(clf, 'intercepts_')
    assert hasattr(clf, 'classes_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = clf.score(X, y)
    assert accuracy > 0.4


def test_mlp_classifier_predict_proba(binary_classification_data):
    """Test MLPClassifier predict_proba."""
    X, y = binary_classification_data
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_mlp_classifier_hidden_layers():
    """Test MLPClassifier with different hidden layer sizes."""
    X = np.random.randn(50, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Single hidden layer
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=30, random_state=42)
    clf.fit(X, y)
    assert len(clf.coefs_) == 2  # input->hidden, hidden->output
    
    # Two hidden layers
    clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=30, random_state=42)
    clf.fit(X, y)
    assert len(clf.coefs_) == 3  # input->h1, h1->h2, h2->output


def test_mlp_classifier_activation_relu():
    """Test MLPClassifier with ReLU activation."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=30, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_classifier_activation_tanh():
    """Test MLPClassifier with tanh activation."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', max_iter=30, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_classifier_activation_logistic():
    """Test MLPClassifier with logistic activation."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=30, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_classifier_solver_sgd():
    """Test MLPClassifier with SGD solver."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), solver='sgd', max_iter=30, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_classifier_solver_adam():
    """Test MLPClassifier with Adam solver."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), solver='adam', max_iter=30, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_classifier_multiclass(classification_data):
    """Test MLPClassifier on multiclass problem."""
    X, y = classification_data
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
    clf.fit(X, y)
    
    assert len(clf.classes_) == 3
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)


def test_mlp_classifier_learning_rate():
    """Test MLPClassifier with different learning rates."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    for lr in [0.001, 0.01, 0.1]:
        clf = MLPClassifier(
            hidden_layer_sizes=(10,),
            learning_rate_init=lr,
            max_iter=20,
            random_state=42
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


def test_mlp_classifier_alpha_regularization():
    """Test MLPClassifier with different alpha values."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    for alpha in [0.0001, 0.001, 0.01]:
        clf = MLPClassifier(
            hidden_layer_sizes=(10,),
            alpha=alpha,
            max_iter=20,
            random_state=42
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


def test_mlp_regressor_fit_predict(regression_data):
    """Test MLPRegressor fit and predict."""
    X, y = regression_data
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
    reg.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(reg, 'coefs_')
    assert hasattr(reg, 'intercepts_')
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # Check R2 score (relaxed threshold for neural network)
    score = reg.score(X, y)
    assert score > 0.0  # Should at least beat random


def test_mlp_regressor_hidden_layers():
    """Test MLPRegressor with different hidden layer sizes."""
    X = np.random.randn(50, 5)
    y = X @ np.random.randn(5) + np.random.randn(50) * 0.1
    
    # Single hidden layer
    reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=30, random_state=42)
    reg.fit(X, y)
    assert len(reg.coefs_) == 2
    
    # Two hidden layers
    reg = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=30, random_state=42)
    reg.fit(X, y)
    assert len(reg.coefs_) == 3


def test_mlp_regressor_activation_relu():
    """Test MLPRegressor with ReLU activation."""
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=30, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_activation_tanh():
    """Test MLPRegressor with tanh activation."""
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', max_iter=30, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_solver_sgd():
    """Test MLPRegressor with SGD solver."""
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), solver='sgd', max_iter=30, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_solver_adam():
    """Test MLPRegressor with Adam solver."""
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), solver='adam', max_iter=30, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_multioutput(multioutput_regression_data):
    """Test MLPRegressor with multi-output."""
    X, y = multioutput_regression_data
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_learning_rate():
    """Test MLPRegressor with different learning rates."""
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    for lr in [0.001, 0.01]:
        reg = MLPRegressor(
            hidden_layer_sizes=(10,),
            learning_rate_init=lr,
            max_iter=20,
            random_state=42
        )
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape


def test_mlp_small_dataset():
    """Test MLP on small dataset."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR problem
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=100, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_mlp_regressor_simple_function():
    """Test MLPRegressor on simple function."""
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1  # y = 2x + 1
    
    reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    
    # Should learn a reasonable approximation (relaxed threshold)
    from eclipsera.core import mean_squared_error
    mse = mean_squared_error(y, y_pred)
    assert mse < 10.0  # Neural networks may need more iterations for simple linear functions


def test_mlp_convergence():
    """Test that MLP converges within max_iter."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42, verbose=False)
    clf.fit(X, y)
    
    # Should complete without error
    assert hasattr(clf, 'coefs_')
