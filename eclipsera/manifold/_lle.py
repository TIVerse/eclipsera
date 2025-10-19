"""Locally Linear Embedding."""
from typing import Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class LocallyLinearEmbedding(BaseEstimator):
    """Locally Linear Embedding.
    
    Non-linear dimensionality reduction that seeks a lower-dimensional
    projection of the data which preserves distances within local neighborhoods.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    reg : float, default=1e-3
        Regularization constant.
        
    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.
    reconstruction_error_ : float
        Reconstruction error.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.manifold import LocallyLinearEmbedding
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> embedding = LocallyLinearEmbedding(n_components=1)
    >>> X_transformed = embedding.fit_transform(X)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        reg: float = 1e-3,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
    
    def _barycenter_weights(self, X: np.ndarray, neighbors: np.ndarray, reg: float) -> np.ndarray:
        """Compute barycenter weights of X from its neighbors."""
        n_neighbors = neighbors.shape[0]
        
        # Center neighbors
        z = X[neighbors] - X
        
        # Local covariance
        C = z @ z.T
        
        # Regularization
        trace = np.trace(C)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        
        C.flat[::n_neighbors + 1] += R
        
        # Solve for weights
        w = np.linalg.solve(C, np.ones(n_neighbors))
        w /= w.sum()
        
        return w
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and transform data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        X = check_array(X)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors must be less than n_samples, "
                f"got n_neighbors={self.n_neighbors}, n_samples={n_samples}"
            )
        
        # Step 1: Find neighbors
        # Compute pairwise distances
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Step 2: Compute reconstruction weights
        W = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            neighbors = np.argsort(distances[i])[1:self.n_neighbors + 1]
            weights = self._barycenter_weights(X[i], neighbors, self.reg)
            W[i, neighbors] = weights
        
        # Step 3: Compute embedding
        # M = (I - W)^T (I - W)
        M = np.eye(n_samples) - W
        M = M.T @ M
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # Select eigenvectors corresponding to smallest eigenvalues (skip first)
        idx = np.argsort(eigenvalues)[1:self.n_components + 1]
        self.embedding_ = eigenvectors[:, idx]
        
        # Reconstruction error
        self.reconstruction_error_ = eigenvalues[1:self.n_components + 1].sum()
        
        return self.embedding_
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LocallyLinearEmbedding":
        """Fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : LocallyLinearEmbedding
            Fitted estimator.
        """
        self.fit_transform(X, y)
        return self
