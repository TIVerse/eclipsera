"""Gaussian Mixture Model."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.utils import check_random_state
from ..core.validation import check_array


class GaussianMixture(BaseEstimator):
    """Gaussian Mixture Model.
    
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution using the Expectation-Maximization (EM) algorithm.
    
    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'diag', 'spherical'}, default='full'
        Type of covariance parameters to use.
    max_iter : int, default=100
        The number of EM iterations to perform.
    tol : float, default=1e-3
        The convergence threshold.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed.
        
    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        The weights of each mixture component.
    means_ : ndarray of shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : ndarray
        The covariance of each mixture component.
    converged_ : bool
        True if converged, False otherwise.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0)
    >>> gm.fit(X)
    GaussianMixture(n_components=2, random_state=0)
    >>> gm.predict([[0, 0], [12, 3]])
    array([0, 1])
    """
    
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: Literal["full", "diag", "spherical"] = "full",
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def _initialize_parameters(self, X: np.ndarray, random_state: np.random.RandomState):
        """Initialize GMM parameters."""
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using random samples
        indices = random_state.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # Initialize covariances
        if self.covariance_type == "full":
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == "diag":
            self.covariances_ = np.ones((self.n_components, n_features))
        else:  # spherical
            self.covariances_ = np.ones(self.n_components)
    
    def _compute_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Compute log probability of X under each component."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            
            if self.covariance_type == "full":
                cov = self.covariances_[k]
                cov_det = np.linalg.det(cov) + 1e-6
                cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
                
                log_prob[:, k] = -0.5 * (
                    np.log(cov_det) +
                    np.sum(diff @ cov_inv * diff, axis=1) +
                    X.shape[1] * np.log(2 * np.pi)
                )
            elif self.covariance_type == "diag":
                cov = self.covariances_[k] + 1e-6
                log_prob[:, k] = -0.5 * (
                    np.sum(np.log(cov)) +
                    np.sum(diff**2 / cov, axis=1) +
                    X.shape[1] * np.log(2 * np.pi)
                )
            else:  # spherical
                cov = self.covariances_[k] + 1e-6
                log_prob[:, k] = -0.5 * (
                    X.shape[1] * np.log(cov) +
                    np.sum(diff**2, axis=1) / cov +
                    X.shape[1] * np.log(2 * np.pi)
                )
        
        return log_prob
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: compute responsibilities."""
        log_prob = self._compute_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-10)
        
        log_resp = log_prob + log_weights
        log_resp -= np.max(log_resp, axis=1, keepdims=True)  # Numerical stability
        
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        
        return resp
    
    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """M-step: update parameters."""
        n_samples, n_features = X.shape
        
        # Update weights
        nk = resp.sum(axis=0) + 1e-10
        self.weights_ = nk / n_samples
        
        # Update means
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]
        
        # Update covariances
        if self.covariance_type == "full":
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k, np.newaxis] * diff).T @ diff / nk[k]
                self.covariances_[k] += 1e-6 * np.eye(n_features)
        
        elif self.covariance_type == "diag":
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k, np.newaxis] * diff**2, axis=0) / nk[k]
        
        else:  # spherical
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k] * np.sum(diff**2, axis=1)) / (nk[k] * n_features)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GaussianMixture":
        """Estimate model parameters with the EM algorithm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : GaussianMixture
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        random_state = check_random_state(self.random_state)
        
        # Initialize parameters
        self._initialize_parameters(X, random_state)
        
        prev_log_likelihood = -np.inf
        self.converged_ = False
        
        # EM algorithm
        for iteration in range(self.max_iter):
            # E-step
            resp = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
            
            # Check convergence
            log_prob = self._compute_log_prob(X)
            log_likelihood = np.sum(np.log(np.sum(np.exp(log_prob) * self.weights_, axis=1)))
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
        X = check_array(X)
        
        resp = self._e_step(X)
        return np.argmax(resp, axis=1)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
        self.fit(X, y)
        return self.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each component.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.
            
        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Posterior probabilities.
        """
        X = check_array(X)
        
        return self._e_step(X)
