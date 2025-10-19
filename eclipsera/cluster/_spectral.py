"""Spectral clustering."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.utils import check_random_state
from ..core.validation import check_array


class SpectralClustering(BaseEstimator):
    """Spectral clustering.
    
    Apply clustering to a projection of the normalized Laplacian.
    Uses the eigenvectors of the graph Laplacian to perform dimensionality
    reduction before clustering in fewer dimensions.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    affinity : {'rbf', 'nearest_neighbors'}, default='rbf'
        How to construct the affinity matrix.
    gamma : float, default=1.0
        Kernel coefficient for rbf kernel.
    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix
        using the nearest neighbors method.
    random_state : int, RandomState instance or None, default=None
        Used for initialization and for reproducibility.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix used for clustering.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import SpectralClustering
    >>> X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralClustering(n_clusters=2, random_state=0)
    >>> clustering.fit(X)
    SpectralClustering(n_clusters=2, random_state=0)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        affinity: Literal["rbf", "nearest_neighbors"] = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def _compute_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute affinity matrix."""
        n_samples = X.shape[0]
        
        if self.affinity == "rbf":
            # RBF (Gaussian) kernel
            pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                               np.sum(X**2, axis=1)[np.newaxis, :] - \
                               2 * X @ X.T
            affinity = np.exp(-self.gamma * pairwise_sq_dists)
        
        elif self.affinity == "nearest_neighbors":
            # K-nearest neighbors
            pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                               np.sum(X**2, axis=1)[np.newaxis, :] - \
                               2 * X @ X.T
            pairwise_dists = np.sqrt(np.maximum(pairwise_sq_dists, 0))
            
            affinity = np.zeros((n_samples, n_samples))
            
            # For each point, connect to k nearest neighbors
            for i in range(n_samples):
                # Get k+1 nearest (including self)
                nearest = np.argsort(pairwise_dists[i])[:self.n_neighbors + 1]
                affinity[i, nearest] = 1.0
                affinity[nearest, i] = 1.0  # Make symmetric
            
            np.fill_diagonal(affinity, 0)  # Remove self-loops
        
        else:
            raise ValueError(f"Unknown affinity: {self.affinity}")
        
        return affinity
    
    def _compute_laplacian(self, affinity: np.ndarray) -> np.ndarray:
        """Compute normalized graph Laplacian."""
        # Degree matrix
        D = np.diag(np.sum(affinity, axis=1))
        
        # Compute D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        
        # Normalized Laplacian: L = D^(-1/2) @ A @ D^(-1/2)
        # Or equivalently: L = I - D^(-1/2) @ A @ D^(-1/2)
        # We want the affinity version for spectral clustering
        L = D_inv_sqrt @ affinity @ D_inv_sqrt
        
        return L
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SpectralClustering":
        """Perform spectral clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : SpectralClustering
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity_matrix(X)
        
        # Compute normalized Laplacian
        L = self._compute_laplacian(self.affinity_matrix_)
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalues (descending for affinity, use top k)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Use top k eigenvectors for clustering
        embedding = eigenvectors[:, :self.n_clusters]
        
        # Normalize rows to unit length
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        embedding = embedding / row_norms
        
        # Apply k-means to the embedding
        from ._kmeans import KMeans
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.labels_ = kmeans.fit_predict(embedding)
        
        return self
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform clustering and return labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_
