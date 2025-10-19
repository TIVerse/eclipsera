"""DBSCAN clustering."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class DBSCAN(BaseEstimator):
    """DBSCAN clustering algorithm.
    
    Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.
    
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances.
    algorithm : str, default='auto'
        The algorithm to be used by the NearestNeighbors module (not implemented).
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are
        given the label -1.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import DBSCAN
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2)
    >>> clustering.fit(X)
    DBSCAN(eps=3, min_samples=2)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        if self.metric == "euclidean":
            # Euclidean distance
            distances = np.sqrt(
                np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
            )
        elif self.metric == "manhattan":
            # Manhattan distance
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def _get_neighbors(self, distances: np.ndarray, point_idx: int) -> np.ndarray:
        """Get neighbors within eps distance."""
        return np.where(distances[point_idx] <= self.eps)[0]
    
    def _expand_cluster(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        point_idx: int,
        neighbors: np.ndarray,
        cluster_id: int,
    ) -> None:
        """Expand cluster from a core point."""
        labels[point_idx] = cluster_id
        
        # Use a set to track points to process
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor is noise, add to cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited
            elif labels[neighbor_idx] == -2:  # Unvisited
                labels[neighbor_idx] = cluster_id
                
                # Get neighbors of neighbor
                neighbor_neighbors = self._get_neighbors(distances, neighbor_idx)
                
                # If neighbor is a core point, add its neighbors to the queue
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
            
            i += 1
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DBSCAN":
        """Perform DBSCAN clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : DBSCAN
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = self._compute_distances(X)
        
        # Initialize labels (-2 = unvisited, -1 = noise, >=0 = cluster id)
        labels = np.full(n_samples, -2, dtype=int)
        
        cluster_id = 0
        core_sample_indices = []
        
        # Process each point
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != -2:
                continue
            
            # Get neighbors
            neighbors = self._get_neighbors(distances, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                # Mark as noise
                labels[point_idx] = -1
            else:
                # Core point - expand cluster
                core_sample_indices.append(point_idx)
                self._expand_cluster(distances, labels, point_idx, neighbors, cluster_id)
                cluster_id += 1
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_sample_indices)
        self.components_ = X[self.core_sample_indices_].copy() if len(core_sample_indices) > 0 else np.array([])
        
        return self
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute clusters and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return self.labels_
