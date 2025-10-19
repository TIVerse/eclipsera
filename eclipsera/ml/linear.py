"""Linear models for regression and classification."""
import warnings
from typing import Optional

import numpy as np
import scipy.linalg as la
from scipy import optimize

from ..core.base import BaseClassifier, BaseRegressor
from ..core.exceptions import ConvergenceWarning
from ..core.utils import safe_sparse_dot, softmax
from ..core.validation import check_array, check_X_y


class LinearRegression(BaseRegressor):
    """Ordinary Least Squares Linear Regression.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : int, default=None
        Number of jobs to use for computation.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients.
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in the linear model.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.linear import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
            
        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        if self.copy_X:
            X = X.copy()
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X = X - X_mean
            y = y - y_mean
        
        # Solve normal equations using SVD for numerical stability
        coef, residuals, rank, s = la.lstsq(X, y)
        
        # For multi-output, transpose to (n_outputs, n_features)
        if y.ndim == 2:
            self.coef_ = coef.T
            if self.fit_intercept:
                self.intercept_ = y_mean - np.dot(X_mean, coef)
            else:
                self.intercept_ = np.zeros(y.shape[1])
        else:
            self.coef_ = coef
            if self.fit_intercept:
                self.intercept_ = y_mean - np.dot(X_mean, coef)
            else:
                self.intercept_ = 0.0
        
        self.rank_ = rank
        self.singular_ = s
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        X = check_array(X)
        if self.coef_.ndim == 2:
            # Multi-output: coef_ is (n_outputs, n_features), so use X @ coef_.T
            return X @ self.coef_.T + self.intercept_
        else:
            return safe_sparse_dot(X, self.coef_) + self.intercept_


class Ridge(BaseRegressor):
    """Ridge regression with L2 regularization."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        solver: str = "auto",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        if self.copy_X:
            X = X.copy()
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X = X - X_mean
            y = y - y_mean
        
        # Solve ridge regression
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        
        self.coef_ = la.solve(A, b, assume_a='pos')
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_) + self.intercept_


class Lasso(BaseRegressor):
    """Lasso regression with L1 regularization."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        X, y = check_X_y(X, y, y_numeric=True)
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        
        # Coordinate descent for Lasso
        self.coef_ = self._coordinate_descent(X, y)
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        coef = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            for j in range(n_features):
                residual = y - X @ coef + X[:, j] * coef[j]
                rho = X[:, j] @ residual
                
                if rho < -self.alpha:
                    coef[j] = (rho + self.alpha) / (X[:, j] @ X[:, j])
                elif rho > self.alpha:
                    coef[j] = (rho - self.alpha) / (X[:, j] @ X[:, j])
                else:
                    coef[j] = 0.0
            
            if np.sum(np.abs(coef - coef_old)) < self.tol:
                break
        
        return coef
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_) + self.intercept_


class ElasticNet(BaseRegressor):
    """ElasticNet regression with L1 and L2 regularization."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        X, y = check_X_y(X, y, y_numeric=True)
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        
        self.coef_ = self._coordinate_descent(X, y)
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        coef = np.zeros(n_features)
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)
        
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            for j in range(n_features):
                residual = y - X @ coef + X[:, j] * coef[j]
                rho = X[:, j] @ residual
                z = X[:, j] @ X[:, j] + l2_reg
                
                if rho < -l1_reg:
                    coef[j] = (rho + l1_reg) / z
                elif rho > l1_reg:
                    coef[j] = (rho - l1_reg) / z
                else:
                    coef[j] = 0.0
            
            if np.sum(np.abs(coef - coef_old)) < self.tol:
                break
        
        return coef
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_) + self.intercept_


class LogisticRegression(BaseClassifier):
    """Logistic Regression classifier."""
    
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        solver: str = "lbfgs",
    ):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Convert labels to indices
        y_encoded = np.searchsorted(self.classes_, y)
        
        if self.n_classes_ == 2:
            # Binary classification
            self.coef_, self.intercept_ = self._fit_binary(X, y_encoded)
        else:
            # Multiclass classification (one-vs-rest)
            self.coef_ = np.zeros((self.n_classes_, X.shape[1]))
            self.intercept_ = np.zeros(self.n_classes_)
            
            for i in range(self.n_classes_):
                y_binary = (y_encoded == i).astype(int)
                coef, intercept = self._fit_binary(X, y_binary)
                self.coef_[i] = coef
                self.intercept_[i] = intercept
        
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> tuple:
        n_features = X.shape[1]
        w0 = np.zeros(n_features + 1 if self.fit_intercept else n_features)
        
        def loss_and_grad(w):
            if self.fit_intercept:
                coef, intercept = w[:-1], w[-1]
            else:
                coef, intercept = w, 0.0
            
            z = X @ coef + intercept
            prob = 1 / (1 + np.exp(-z))
            
            # Cross-entropy loss
            loss = -np.mean(y * np.log(prob + 1e-15) + (1 - y) * np.log(1 - prob + 1e-15))
            
            # L2 regularization
            if self.penalty == "l2":
                loss += (1 / (2 * self.C)) * np.sum(coef ** 2)
            
            # Gradient
            grad_coef = X.T @ (prob - y) / len(y)
            if self.penalty == "l2":
                grad_coef += coef / self.C
            
            if self.fit_intercept:
                grad_intercept = np.mean(prob - y)
                return loss, np.concatenate([grad_coef, [grad_intercept]])
            else:
                return loss, grad_coef
        
        result = optimize.minimize(
            loss_and_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge", ConvergenceWarning)
        
        if self.fit_intercept:
            return result.x[:-1], result.x[-1]
        else:
            return result.x, 0.0
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        
        if self.n_classes_ == 2:
            z = X @ self.coef_ + self.intercept_
            prob_class1 = 1 / (1 + np.exp(-z))
            return np.column_stack([1 - prob_class1, prob_class1])
        else:
            z = X @ self.coef_.T + self.intercept_
            return softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
