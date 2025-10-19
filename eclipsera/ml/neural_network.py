"""Multi-layer Perceptron (Neural Network) for classification and regression."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseClassifier, BaseRegressor
from ..core.exceptions import ConvergenceWarning, NotFittedError
from ..core.utils import check_random_state, softmax
from ..core.validation import check_array, check_X_y


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def logistic_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of logistic sigmoid."""
    s = logistic(x)
    return s * (1 - s)


class MLPClassifier(BaseClassifier):
    """Multi-layer Perceptron classifier.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'relu', 'tanh', 'logistic'}, default='relu'
        Activation function for the hidden layer.
    solver : {'sgd', 'adam'}, default='adam'
        The solver for weight optimization.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
    learning_rate_init : float, default=0.001
        The initial learning rate used.
    max_iter : int, default=200
        Maximum number of iterations.
    random_state : int, RandomState instance or None, default=None
        Random number generation for weights initialization.
    tol : float, default=1e-4
        Tolerance for the optimization.
    verbose : bool, default=False
        Whether to print progress messages.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.
    coefs_ : list of ndarray
        The ith element represents the weight matrix for layer i.
    intercepts_ : list of ndarray
        The ith element represents the bias vector for layer i + 1.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: Literal["relu", "tanh", "logistic"] = "relu",
        solver: Literal["sgd", "adam"] = "adam",
        alpha: float = 0.0001,
        batch_size: int = 200,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
    
    def _initialize_weights(self, layer_sizes: list) -> None:
        """Initialize weights and biases."""
        random_state = check_random_state(self.random_state)
        
        self.coefs_ = []
        self.intercepts_ = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            
            coef = random_state.uniform(-limit, limit, (fan_in, fan_out))
            intercept = random_state.uniform(-limit, limit, fan_out)
            
            self.coefs_.append(coef)
            self.intercepts_.append(intercept)
    
    def _forward_pass(self, X: np.ndarray) -> tuple:
        """Perform a forward pass through the network."""
        activations = [X]
        z_values = []
        
        for i in range(len(self.coefs_)):
            z = activations[-1] @ self.coefs_[i] + self.intercepts_[i]
            z_values.append(z)
            
            if i < len(self.coefs_) - 1:
                # Hidden layers
                if self.activation == "relu":
                    activation = relu(z)
                elif self.activation == "tanh":
                    activation = np.tanh(z)
                elif self.activation == "logistic":
                    activation = logistic(z)
            else:
                # Output layer (softmax for classification)
                activation = softmax(z)
            
            activations.append(activation)
        
        return activations, z_values
    
    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: list,
        z_values: list
    ) -> tuple:
        """Perform backpropagation to compute gradients."""
        n_samples = X.shape[0]
        
        # Initialize gradients
        coef_grads = [np.zeros_like(coef) for coef in self.coefs_]
        intercept_grads = [np.zeros_like(intercept) for intercept in self.intercepts_]
        
        # Output layer error (cross-entropy with softmax)
        delta = activations[-1] - y
        
        # Backpropagate
        for i in range(len(self.coefs_) - 1, -1, -1):
            coef_grads[i] = activations[i].T @ delta / n_samples + \
                           self.alpha * self.coefs_[i] / n_samples
            intercept_grads[i] = np.mean(delta, axis=0)
            
            if i > 0:
                # Propagate error to previous layer
                delta = (delta @ self.coefs_[i].T)
                
                # Apply activation derivative
                if self.activation == "relu":
                    delta *= relu_derivative(z_values[i - 1])
                elif self.activation == "tanh":
                    delta *= tanh_derivative(z_values[i - 1])
                elif self.activation == "logistic":
                    delta *= logistic_derivative(z_values[i - 1])
        
        return coef_grads, intercept_grads
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Fit the MLP classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : MLPClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Convert labels to one-hot encoding
        y_onehot = np.zeros((len(y), self.n_classes_))
        for idx, c in enumerate(self.classes_):
            y_onehot[y == c, idx] = 1
        
        # Determine layer sizes
        layer_sizes = [self.n_features_in_] + list(self.hidden_layer_sizes) + [self.n_classes_]
        
        # Initialize weights
        self._initialize_weights(layer_sizes)
        
        # Initialize Adam optimizer variables if needed
        if self.solver == "adam":
            self.m_coefs_ = [np.zeros_like(coef) for coef in self.coefs_]
            self.v_coefs_ = [np.zeros_like(coef) for coef in self.coefs_]
            self.m_intercepts_ = [np.zeros_like(intercept) for intercept in self.intercepts_]
            self.v_intercepts_ = [np.zeros_like(intercept) for intercept in self.intercepts_]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
        
        # Training loop
        learning_rate = self.learning_rate_init
        best_loss = np.inf
        no_improvement = 0
        
        for iteration in range(self.max_iter):
            # Forward pass
            activations, z_values = self._forward_pass(X)
            
            # Compute loss
            predictions = activations[-1]
            loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-10), axis=1))
            loss += 0.5 * self.alpha * sum(np.sum(coef ** 2) for coef in self.coefs_) / len(X)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, loss: {loss:.4f}")
            
            # Check for convergence
            if loss < best_loss - self.tol:
                best_loss = loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= 10:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Backward pass
            coef_grads, intercept_grads = self._backward_pass(X, y_onehot, activations, z_values)
            
            # Update weights
            if self.solver == "sgd":
                for i in range(len(self.coefs_)):
                    self.coefs_[i] -= learning_rate * coef_grads[i]
                    self.intercepts_[i] -= learning_rate * intercept_grads[i]
            
            elif self.solver == "adam":
                t = iteration + 1
                for i in range(len(self.coefs_)):
                    # Update biased first moment estimate
                    self.m_coefs_[i] = self.beta1 * self.m_coefs_[i] + \
                                      (1 - self.beta1) * coef_grads[i]
                    self.m_intercepts_[i] = self.beta1 * self.m_intercepts_[i] + \
                                           (1 - self.beta1) * intercept_grads[i]
                    
                    # Update biased second raw moment estimate
                    self.v_coefs_[i] = self.beta2 * self.v_coefs_[i] + \
                                      (1 - self.beta2) * (coef_grads[i] ** 2)
                    self.v_intercepts_[i] = self.beta2 * self.v_intercepts_[i] + \
                                           (1 - self.beta2) * (intercept_grads[i] ** 2)
                    
                    # Compute bias-corrected moment estimates
                    m_hat_coef = self.m_coefs_[i] / (1 - self.beta1 ** t)
                    m_hat_intercept = self.m_intercepts_[i] / (1 - self.beta1 ** t)
                    v_hat_coef = self.v_coefs_[i] / (1 - self.beta2 ** t)
                    v_hat_intercept = self.v_intercepts_[i] / (1 - self.beta2 ** t)
                    
                    # Update parameters
                    self.coefs_[i] -= learning_rate * m_hat_coef / \
                                     (np.sqrt(v_hat_coef) + self.epsilon)
                    self.intercepts_[i] -= learning_rate * m_hat_intercept / \
                                          (np.sqrt(v_hat_intercept) + self.epsilon)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X = check_array(X)
        
        if not hasattr(self, 'coefs_'):
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the MLP classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class MLPRegressor(BaseRegressor):
    """Multi-layer Perceptron regressor.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'relu', 'tanh', 'logistic'}, default='relu'
        Activation function for the hidden layer.
    solver : {'sgd', 'adam'}, default='adam'
        The solver for weight optimization.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    learning_rate_init : float, default=0.001
        The initial learning rate used.
    max_iter : int, default=200
        Maximum number of iterations.
    random_state : int, RandomState instance or None, default=None
        Random number generation for weights initialization.
    tol : float, default=1e-4
        Tolerance for the optimization.
    verbose : bool, default=False
        Whether to print progress messages.
        
    Attributes
    ----------
    coefs_ : list of ndarray
        The ith element represents the weight matrix for layer i.
    intercepts_ : list of ndarray
        The ith element represents the bias vector for layer i + 1.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: Literal["relu", "tanh", "logistic"] = "relu",
        solver: Literal["sgd", "adam"] = "adam",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
    
    def _initialize_weights(self, layer_sizes: list) -> None:
        """Initialize weights and biases."""
        random_state = check_random_state(self.random_state)
        
        self.coefs_ = []
        self.intercepts_ = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            
            coef = random_state.uniform(-limit, limit, (fan_in, fan_out))
            intercept = random_state.uniform(-limit, limit, fan_out)
            
            self.coefs_.append(coef)
            self.intercepts_.append(intercept)
    
    def _forward_pass(self, X: np.ndarray) -> tuple:
        """Perform a forward pass through the network."""
        activations = [X]
        z_values = []
        
        for i in range(len(self.coefs_)):
            z = activations[-1] @ self.coefs_[i] + self.intercepts_[i]
            z_values.append(z)
            
            if i < len(self.coefs_) - 1:
                # Hidden layers
                if self.activation == "relu":
                    activation = relu(z)
                elif self.activation == "tanh":
                    activation = np.tanh(z)
                elif self.activation == "logistic":
                    activation = logistic(z)
            else:
                # Output layer (identity for regression)
                activation = z
            
            activations.append(activation)
        
        return activations, z_values
    
    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: list,
        z_values: list
    ) -> tuple:
        """Perform backpropagation to compute gradients."""
        n_samples = X.shape[0]
        
        coef_grads = [np.zeros_like(coef) for coef in self.coefs_]
        intercept_grads = [np.zeros_like(intercept) for intercept in self.intercepts_]
        
        # Output layer error (MSE)
        delta = activations[-1] - y
        
        # Backpropagate
        for i in range(len(self.coefs_) - 1, -1, -1):
            coef_grads[i] = activations[i].T @ delta / n_samples + \
                           self.alpha * self.coefs_[i] / n_samples
            intercept_grads[i] = np.mean(delta, axis=0)
            
            if i > 0:
                delta = (delta @ self.coefs_[i].T)
                
                if self.activation == "relu":
                    delta *= relu_derivative(z_values[i - 1])
                elif self.activation == "tanh":
                    delta *= tanh_derivative(z_values[i - 1])
                elif self.activation == "logistic":
                    delta *= logistic_derivative(z_values[i - 1])
        
        return coef_grads, intercept_grads
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Fit the MLP regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
            
        Returns
        -------
        self : MLPRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        self.n_features_in_ = X.shape[1]
        n_outputs = y.shape[1] if y.ndim == 2 else 1
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Determine layer sizes
        layer_sizes = [self.n_features_in_] + list(self.hidden_layer_sizes) + [n_outputs]
        
        # Initialize weights
        self._initialize_weights(layer_sizes)
        
        # Initialize Adam optimizer if needed
        if self.solver == "adam":
            self.m_coefs_ = [np.zeros_like(coef) for coef in self.coefs_]
            self.v_coefs_ = [np.zeros_like(coef) for coef in self.coefs_]
            self.m_intercepts_ = [np.zeros_like(intercept) for intercept in self.intercepts_]
            self.v_intercepts_ = [np.zeros_like(intercept) for intercept in self.intercepts_]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
        
        # Training loop
        learning_rate = self.learning_rate_init
        best_loss = np.inf
        no_improvement = 0
        
        for iteration in range(self.max_iter):
            activations, z_values = self._forward_pass(X)
            
            # Compute MSE loss
            predictions = activations[-1]
            loss = np.mean((predictions - y) ** 2)
            loss += 0.5 * self.alpha * sum(np.sum(coef ** 2) for coef in self.coefs_) / len(X)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, loss: {loss:.4f}")
            
            if loss < best_loss - self.tol:
                best_loss = loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= 10:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            coef_grads, intercept_grads = self._backward_pass(X, y, activations, z_values)
            
            # Update weights (SGD or Adam)
            if self.solver == "sgd":
                for i in range(len(self.coefs_)):
                    self.coefs_[i] -= learning_rate * coef_grads[i]
                    self.intercepts_[i] -= learning_rate * intercept_grads[i]
            
            elif self.solver == "adam":
                t = iteration + 1
                for i in range(len(self.coefs_)):
                    self.m_coefs_[i] = self.beta1 * self.m_coefs_[i] + \
                                      (1 - self.beta1) * coef_grads[i]
                    self.m_intercepts_[i] = self.beta1 * self.m_intercepts_[i] + \
                                           (1 - self.beta1) * intercept_grads[i]
                    
                    self.v_coefs_[i] = self.beta2 * self.v_coefs_[i] + \
                                      (1 - self.beta2) * (coef_grads[i] ** 2)
                    self.v_intercepts_[i] = self.beta2 * self.v_intercepts_[i] + \
                                           (1 - self.beta2) * (intercept_grads[i] ** 2)
                    
                    m_hat_coef = self.m_coefs_[i] / (1 - self.beta1 ** t)
                    m_hat_intercept = self.m_intercepts_[i] / (1 - self.beta1 ** t)
                    v_hat_coef = self.v_coefs_[i] / (1 - self.beta2 ** t)
                    v_hat_intercept = self.v_intercepts_[i] / (1 - self.beta2 ** t)
                    
                    self.coefs_[i] -= learning_rate * m_hat_coef / \
                                     (np.sqrt(v_hat_coef) + self.epsilon)
                    self.intercepts_[i] -= learning_rate * m_hat_intercept / \
                                          (np.sqrt(v_hat_intercept) + self.epsilon)
        
        self.n_outputs_ = n_outputs
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the MLP regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'coefs_'):
            raise NotFittedError("This MLPRegressor instance is not fitted yet.")
        
        activations, _ = self._forward_pass(X)
        predictions = activations[-1]
        
        if self.n_outputs_ == 1:
            return predictions.ravel()
        
        return predictions
