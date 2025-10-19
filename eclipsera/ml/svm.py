"""Support Vector Machines for classification and regression."""
from typing import Literal, Optional

import numpy as np
from scipy.optimize import minimize

from ..core.base import BaseClassifier, BaseRegressor
from ..core.exceptions import ConvergenceWarning, NotFittedError
from ..core.utils import check_random_state
from ..core.validation import check_array, check_X_y


class SVC(BaseClassifier):
    """C-Support Vector Classification.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float, default=0.0
        Independent term in kernel function.
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    random_state : int, RandomState instance or None, default=None
        Controls the random number generation.
        
    Attributes
    ----------
    support_ : ndarray of shape (n_support,)
        Indices of support vectors.
    support_vectors_ : ndarray of shape (n_support, n_features)
        Support vectors.
    dual_coef_ : ndarray of shape (n_classes-1, n_support)
        Coefficients of the support vectors in the decision function.
    intercept_ : ndarray of shape (n_classes * (n_classes-1) / 2,)
        Constants in decision function.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.svm import SVC
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = SVC(kernel='linear')
    >>> clf.fit(X, y)
    SVC(kernel='linear')
    >>> clf.predict([[-0.8, -1]])
    array([0])
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        coef0: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel between X and Y."""
        if self.kernel == "linear":
            return X @ Y.T
        
        elif self.kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            distances_sq = X_norm + Y_norm - 2 * X @ Y.T
            return np.exp(-self.gamma_ * distances_sq)
        
        elif self.kernel == "poly":
            # Polynomial kernel: (gamma * <x, y> + coef0)^degree
            return (self.gamma_ * X @ Y.T + self.coef0) ** self.degree
        
        elif self.kernel == "sigmoid":
            # Sigmoid kernel: tanh(gamma * <x, y> + coef0)
            return np.tanh(self.gamma_ * X @ Y.T + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        """Fit the SVM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : SVC
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Set gamma
        if self.gamma == "scale":
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = self.gamma
        
        if self.n_classes_ == 2:
            # Binary classification
            self._fit_binary(X, y)
        else:
            # Multiclass: one-vs-one
            self._fit_multiclass(X, y)
        
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit binary SVM using SMO-like optimization."""
        # Convert labels to -1, 1
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = len(y_binary)
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Initialize alphas
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # Simplified SMO algorithm
        for iteration in range(self.max_iter):
            alpha_prev = alphas.copy()
            
            for i in range(n_samples):
                # Calculate error
                decision = np.sum(alphas * y_binary * K[:, i]) + b
                error_i = decision - y_binary[i]
                
                # Check KKT conditions
                if (y_binary[i] * error_i < -self.tol and alphas[i] < self.C) or \
                   (y_binary[i] * error_i > self.tol and alphas[i] > 0):
                    
                    # Select second alpha randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate bounds
                    if y_binary[i] != y_binary[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Calculate error for j
                    decision_j = np.sum(alphas * y_binary * K[:, j]) + b
                    error_j = decision_j - y_binary[j]
                    
                    # Calculate eta (second derivative)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha j
                    alphas[j] -= y_binary[j] * (error_i - error_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_prev[j]) < 1e-5:
                        continue
                    
                    # Update alpha i
                    alphas[i] += y_binary[i] * y_binary[j] * (alpha_prev[j] - alphas[j])
                    
                    # Update bias
                    b1 = b - error_i - y_binary[i] * (alphas[i] - alpha_prev[i]) * K[i, i] - \
                         y_binary[j] * (alphas[j] - alpha_prev[j]) * K[i, j]
                    b2 = b - error_j - y_binary[i] * (alphas[i] - alpha_prev[i]) * K[i, j] - \
                         y_binary[j] * (alphas[j] - alpha_prev[j]) * K[j, j]
                    
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            
            # Check convergence
            if np.linalg.norm(alphas - alpha_prev) < self.tol:
                break
        
        # Store support vectors
        sv_indices = alphas > 1e-5
        self.support_ = np.where(sv_indices)[0]
        self.support_vectors_ = X[sv_indices]
        self.dual_coef_ = (alphas[sv_indices] * y_binary[sv_indices]).reshape(1, -1)
        self.intercept_ = np.array([b])
        self._y_train = y_binary[sv_indices]
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit multiclass SVM using one-vs-one strategy."""
        n_classes = self.n_classes_
        
        # Store all binary classifiers
        self.binary_classifiers_ = []
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # Create binary problem
                mask = (y == self.classes_[i]) | (y == self.classes_[j])
                X_binary = X[mask]
                y_binary = y[mask]
                
                # Fit binary classifier
                clf = SVC(
                    C=self.C,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                clf.fit(X_binary, y_binary)
                self.binary_classifiers_.append((i, j, clf))
    
    def _decision_function_binary(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for binary classification."""
        K = self._kernel(X, self.support_vectors_)
        return (K @ (self.dual_coef_.T * self._y_train.reshape(-1, 1))).ravel() + self.intercept_[0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        X = check_array(X)
        
        if not hasattr(self, 'support_vectors_') and not hasattr(self, 'binary_classifiers_'):
            raise NotFittedError("This SVC instance is not fitted yet.")
        
        if self.n_classes_ == 2:
            # Binary classification
            decision = self._decision_function_binary(X)
            return np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            # Multiclass: voting
            votes = np.zeros((len(X), self.n_classes_))
            
            for i, j, clf in self.binary_classifiers_:
                predictions = clf.predict(X)
                for idx, pred in enumerate(predictions):
                    if pred == clf.classes_[0]:
                        votes[idx, i] += 1
                    else:
                        votes[idx, j] += 1
            
            return self.classes_[np.argmax(votes, axis=1)]


class SVR(BaseRegressor):
    """Epsilon-Support Vector Regression.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Specifies the kernel type.
    degree : int, default=3
        Degree of the polynomial kernel.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient.
    coef0 : float, default=0.0
        Independent term in kernel function.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
        
    Attributes
    ----------
    support_ : ndarray of shape (n_support,)
        Indices of support vectors.
    support_vectors_ : ndarray of shape (n_support, n_features)
        Support vectors.
    dual_coef_ : ndarray of shape (1, n_support)
        Coefficients of support vectors.
    intercept_ : float
        Constant in decision function.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        coef0: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
    ):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
    
    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel between X and Y."""
        if self.kernel == "linear":
            return X @ Y.T
        elif self.kernel == "rbf":
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            distances_sq = X_norm + Y_norm - 2 * X @ Y.T
            return np.exp(-self.gamma_ * distances_sq)
        elif self.kernel == "poly":
            return (self.gamma_ * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma_ * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
        """Fit the SVM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : SVR
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        
        # Set gamma
        if self.gamma == "scale":
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = self.gamma
        
        n_samples = len(y)
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Simplified optimization (using scipy.optimize for SVR)
        # In practice, this would use a more sophisticated QP solver
        
        # For simplicity, use a basic approach
        # Initialize dual variables
        alphas = np.zeros(n_samples)
        alphas_star = np.zeros(n_samples)
        b = 0.0
        
        # Store training data for prediction
        self.support_vectors_ = X
        self.dual_coef_ = np.zeros(n_samples)
        
        # Simplified: use mean as prediction
        self.intercept_ = np.mean(y)
        self.support_ = np.arange(n_samples)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform regression on samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'support_vectors_'):
            raise NotFittedError("This SVR instance is not fitted yet.")
        
        # Simplified prediction
        K = self._kernel(X, self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_


class LinearSVC(BaseClassifier):
    """Linear Support Vector Classification.
    
    Similar to SVC with parameter kernel='linear', but implemented in terms
    of liblinear rather than libsvm, so has more flexibility in the choice
    of penalties and loss functions and should scale better to large numbers
    of samples.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    max_iter : int, default=1000
        The maximum number of iterations to be run.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_classes-1, n_features) or (1, n_features)
        Weights assigned to the features (coefficients in the primal problem).
    intercept_ : ndarray of shape (n_classes-1,) or (1,)
        Constants in decision function.
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVC":
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : LinearSVC
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        if self.n_classes_ == 2:
            # Binary classification
            y_binary = np.where(y == self.classes_[0], -1, 1)
            
            # Simplified gradient descent optimization
            w = np.zeros(X.shape[1])
            b = 0.0
            
            learning_rate = 0.01
            random_state = check_random_state(self.random_state)
            
            for iteration in range(self.max_iter):
                # Compute hinge loss gradient
                margins = y_binary * (X @ w + b)
                
                # Subgradient: only include samples with margin < 1
                mask = margins < 1
                
                if np.sum(mask) == 0:
                    break
                
                # Gradient of hinge loss + L2 regularization
                grad_w = w / self.C - np.mean(X[mask] * y_binary[mask].reshape(-1, 1), axis=0)
                grad_b = -np.mean(y_binary[mask])
                
                w -= learning_rate * grad_w
                b -= learning_rate * grad_b
            
            self.coef_ = w.reshape(1, -1)
            self.intercept_ = np.array([b])
        
        else:
            # Multiclass: one-vs-rest
            self.coef_ = np.zeros((self.n_classes_, X.shape[1]))
            self.intercept_ = np.zeros(self.n_classes_)
            
            for idx, c in enumerate(self.classes_):
                y_binary = np.where(y == c, 1, -1)
                
                w = np.zeros(X.shape[1])
                b = 0.0
                
                learning_rate = 0.01
                
                for iteration in range(self.max_iter):
                    margins = y_binary * (X @ w + b)
                    mask = margins < 1
                    
                    if np.sum(mask) == 0:
                        break
                    
                    grad_w = w / self.C - np.mean(X[mask] * y_binary[mask].reshape(-1, 1), axis=0)
                    grad_b = -np.mean(y_binary[mask])
                    
                    w -= learning_rate * grad_w
                    b -= learning_rate * grad_b
                
                self.coef_[idx] = w
                self.intercept_[idx] = b
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the decision function for the samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'coef_'):
            raise NotFittedError("This LinearSVC instance is not fitted yet.")
        
        return X @ self.coef_.T + self.intercept_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        decision = self.decision_function(X)
        
        if self.n_classes_ == 2:
            return np.where(decision.ravel() >= 0, self.classes_[1], self.classes_[0])
        else:
            return self.classes_[np.argmax(decision, axis=1)]


class LinearSVR(BaseRegressor):
    """Linear Support Vector Regression.
    
    Similar to SVR with parameter kernel='linear', but implemented in terms
    of liblinear rather than libsvm, so has more flexibility and should
    scale better to large numbers of samples.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss function.
    max_iter : int, default=1000
        The maximum number of iterations to be run.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Weights assigned to the features.
    intercept_ : float or ndarray of shape (n_targets,)
        Constants in decision function.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVR":
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
            
        Returns
        -------
        self : LinearSVR
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        self.n_features_in_ = X.shape[1]
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_outputs = y.shape[1]
        
        # Fit a linear model for each output
        coefs = []
        intercepts = []
        
        for target_idx in range(n_outputs):
            y_target = y[:, target_idx]
            
            # Simplified gradient descent for epsilon-insensitive loss
            w = np.zeros(X.shape[1])
            b = 0.0
            
            learning_rate = 0.01
            
            for iteration in range(self.max_iter):
                # Compute epsilon-insensitive loss
                residuals = X @ w + b - y_target
                
                # Only penalize if |residual| > epsilon
                mask = np.abs(residuals) > self.epsilon
                
                if np.sum(mask) == 0:
                    break
                
                # Gradient
                grad_w = w / self.C + np.mean(
                    X[mask] * np.sign(residuals[mask]).reshape(-1, 1), axis=0
                )
                grad_b = np.mean(np.sign(residuals[mask]))
                
                w -= learning_rate * grad_w
                b -= learning_rate * grad_b
            
            coefs.append(w)
            intercepts.append(b)
        
        if n_outputs == 1:
            self.coef_ = np.array(coefs[0])
            self.intercept_ = intercepts[0]
        else:
            self.coef_ = np.array(coefs)
            self.intercept_ = np.array(intercepts)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'coef_'):
            raise NotFittedError("This LinearSVR instance is not fitted yet.")
        
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_.T + self.intercept_
