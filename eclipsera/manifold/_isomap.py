"""Isomap for manifold learning."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class Isomap(BaseEstimator):
    """Isomap Embedding.
    
    Non-linear dimensionality reduction through Isometric Mapping.
    Seeks a lower-dimensional embedding which maintains geodesic distances
    between all points.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    metric : str, default='euclidean'
        Distance metric to use.
        
    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.manifold import Isomap
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> embedding = Isomap(n_components=1)
    >>> X_transformed = embedding.fit_transform(X)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        metric: Literal["euclidean"] = "euclidean",
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _construct_neighborhood_graph(self, distances: np.ndarray) -> np.ndarray:
        """Construct k-nearest neighbors graph."""
        n_samples = distances.shape[0]
        graph = np.full((n_samples, n_samples), np.inf)
        
        for i in range(n_samples):
            # Find k nearest neighbors
            neighbors = np.argsort(distances[i])[1:self.n_neighbors + 1]
            for j in neighbors:
                graph[i, j] = distances[i, j]
                graph[j, i] = distances[i, j]  # Make symmetric
        
        return graph
    
    def _floyd_warshall(self, graph: np.ndarray) -> np.ndarray:
        """Compute shortest paths using Floyd-Warshall algorithm."""
        n_samples = graph.shape[0]
        dist = graph.copy()
        
        # Set diagonal to 0
        np.fill_diagonal(dist, 0)
        
        # Floyd-Warshall
        for k in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
    
    def _classical_mds(self, distances: np.ndarray) -> np.ndarray:
        """Classical multidimensional scaling."""
        n_samples = distances.shape[0]
        
        # Centering matrix
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        
        # Double center the squared distance matrix
        B = -0.5 * H @ (distances ** 2) @ H
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top k eigenvalues and eigenvectors
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]
        
        # Compute embedding
        embedding = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        
        return embedding
    
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
        
        self.n_features_in_ = X.shape[1]
        
        # Compute pairwise distances
        distances = self._compute_distance_matrix(X)
        
        # Construct neighborhood graph
        graph = self._construct_neighborhood_graph(distances)
        
        # Compute geodesic distances
        geodesic_distances = self._floyd_warshall(graph)
        
        # Apply MDS
        self.embedding_ = self._classical_mds(geodesic_distances)
        
        return self.embedding_
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Isomap":
        """Fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : Isomap
            Fitted estimator.
        """
        self.fit_transform(X, y)
        return self
