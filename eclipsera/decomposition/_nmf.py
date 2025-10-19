"""Non-negative Matrix Factorization."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.utils import check_random_state
from ..core.validation import check_array


class NMF(BaseTransformer):
    """Non-Negative Matrix Factorization (NMF).
    
    Find two non-negative matrices (W, H) whose product approximates the
    non-negative matrix X. This factorization can be used for dimensionality
    reduction, source separation, or topic modeling.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, n_components = min(n_samples, n_features).
    init : {'random', 'nndsvd'}, default='random'
        Method used to initialize the procedure.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    random_state : int, RandomState instance or None, default=None
        Random number generator seed.
    alpha : float, default=0.0
        Regularization parameter.
        
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix H.
    n_components_ : int
        The number of components.
    reconstruction_err_ : float
        Frobenius norm of the matrix difference between the training data and
        the reconstructed data.
    n_iter_ : int
        Actual number of iterations.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.decomposition import NMF
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Literal["random", "nndsvd"] = "random",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        alpha: float = 0.0,
    ):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha
    
    def _initialize_nmf(self, X: np.ndarray, n_components: int) -> tuple:
        """Initialize W and H matrices."""
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        
        if self.init == "random":
            # Random initialization with small positive values
            W = np.abs(random_state.randn(n_samples, n_components))
            H = np.abs(random_state.randn(n_components, n_features))
        
        elif self.init == "nndsvd":
            # Non-Negative Double Singular Value Decomposition
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            
            W = np.zeros((n_samples, n_components))
            H = np.zeros((n_components, n_features))
            
            # First component
            W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
            H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
            
            # Remaining components
            for j in range(1, min(n_components, len(S))):
                x = U[:, j]
                y = Vt[j, :]
                
                # Positive and negative parts
                x_pos = np.maximum(x, 0)
                y_pos = np.maximum(y, 0)
                x_neg = np.abs(np.minimum(x, 0))
                y_neg = np.abs(np.minimum(y, 0))
                
                # Choose larger contribution
                if np.linalg.norm(x_pos) * np.linalg.norm(y_pos) >= \
                   np.linalg.norm(x_neg) * np.linalg.norm(y_neg):
                    W[:, j] = np.sqrt(S[j] * np.linalg.norm(x_pos) ** 2 / np.linalg.norm(y_pos) ** 2) * x_pos
                    H[j, :] = np.sqrt(S[j] * np.linalg.norm(y_pos) ** 2 / np.linalg.norm(x_pos) ** 2) * y_pos
                else:
                    W[:, j] = np.sqrt(S[j] * np.linalg.norm(x_neg) ** 2 / np.linalg.norm(y_neg) ** 2) * x_neg
                    H[j, :] = np.sqrt(S[j] * np.linalg.norm(y_neg) ** 2 / np.linalg.norm(x_neg) ** 2) * y_neg
            
            # Fill remaining with small random values
            if n_components > len(S):
                W[:, len(S):] = np.abs(random_state.randn(n_samples, n_components - len(S))) * 0.01
                H[len(S):, :] = np.abs(random_state.randn(n_components - len(S), n_features)) * 0.01
        
        else:
            raise ValueError(f"Invalid init parameter: {self.init}")
        
        return W, H
    
    def _update_coordinate_descent(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> tuple:
        """Update W and H using coordinate descent (multiplicative update rules)."""
        # Update H
        WtW = W.T @ W
        WtX = W.T @ X
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        H = H * (WtX / (WtW @ H + self.alpha + epsilon))
        
        # Update W
        HHt = H @ H.T
        XHt = X @ H.T
        
        W = W * (XHt / (W @ HHt + self.alpha + epsilon))
        
        return W, H
    
    def _reconstruction_error(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
        """Compute Frobenius norm of reconstruction error."""
        return np.linalg.norm(X - W @ H, 'fro')
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NMF":
        """Learn a NMF model for the data X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be non-negative.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : NMF
            Fitted estimator.
        """
        X = check_array(X)
        
        if np.any(X < 0):
            raise ValueError("NMF requires non-negative input data")
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Determine number of components
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components
        
        self.n_components_ = n_components
        
        # Initialize W and H
        W, H = self._initialize_nmf(X, n_components)
        
        # Iterative updates
        prev_error = np.inf
        
        for iteration in range(self.max_iter):
            # Update matrices
            W, H = self._update_coordinate_descent(X, W, H)
            
            # Compute reconstruction error
            error = self._reconstruction_error(X, W, H)
            
            # Check convergence
            if abs(prev_error - error) < self.tol:
                break
            
            prev_error = error
        
        self.components_ = H
        self.reconstruction_err_ = error
        self.n_iter_ = iteration + 1
        
        # Store W for transform
        self._W = W
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to the basis defined by components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = check_array(X)
        
        if not hasattr(self, 'components_'):
            raise ValueError("NMF must be fitted before calling transform")
        
        if np.any(X < 0):
            raise ValueError("NMF requires non-negative input data")
        
        # Initialize W
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)
        W = np.abs(random_state.randn(n_samples, self.n_components_))
        
        # Fixed H, optimize W
        H = self.components_
        epsilon = 1e-10
        
        for _ in range(100):  # Fixed number of iterations for transform
            HHt = H @ H.T
            XHt = X @ H.T
            W = W * (XHt / (W @ HHt + epsilon))
        
        return W
    
    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        """Transform data back to original space.
        
        Parameters
        ----------
        W : array-like of shape (n_samples, n_components)
            Transformed data.
            
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        if not hasattr(self, 'components_'):
            raise ValueError("NMF must be fitted before calling inverse_transform")
        
        W = check_array(W)
        return W @ self.components_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Learn a NMF model and return the transformed data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y)
        return self._W
