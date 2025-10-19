"""Mean Shift clustering."""
from typing import Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class MeanShift(BaseEstimator):
    """Mean shift clustering.
    
    Mean shift is a centroid-based algorithm that works by updating candidates
    for centroids to be the mean of the points within a given region. These
    candidates are then filtered in a post-processing stage to eliminate
    near-duplicates to form the final set of centroids.
    
    Parameters
    ----------
    bandwidth : float, default=None
        Bandwidth used in the RBF kernel. If None, estimate using
        quantile of pairwise distances.
    max_iter : int, default=300
        Maximum number of iterations per seed point.
    cluster_all : bool, default=True
        If true, all points are clustered, even those orphans that are
        not within any kernel.
        
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    n_iter_ : int
        Maximum number of iterations performed on each seed.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import MeanShift
    >>> X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2)
    >>> clustering.fit(X)
    MeanShift(bandwidth=2)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.cluster_centers_
    array([[1.33..., 0.66...],
           [3.33..., 6.   ]])
    """
    
    def __init__(
        self,
        bandwidth: Optional[float] = None,
        max_iter: int = 300,
        cluster_all: bool = True,
    ):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.cluster_all = cluster_all
    
    def _estimate_bandwidth(self, X: np.ndarray, quantile: float = 0.3) -> float:
        """Estimate bandwidth using quantile of pairwise distances."""
        n_samples = min(X.shape[0], 1000)  # Subsample for efficiency
        
        # Compute pairwise distances for subset
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_subset = X[idx]
        
        pairwise_sq_dists = np.sum(X_subset**2, axis=1)[:, np.newaxis] + \
                           np.sum(X_subset**2, axis=1)[np.newaxis, :] - \
                           2 * X_subset @ X_subset.T
        
        pairwise_dists = np.sqrt(np.maximum(pairwise_sq_dists, 0))
        
        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(pairwise_dists, k=1)
        distances = pairwise_dists[triu_indices]
        
        # Return quantile
        return np.quantile(distances, quantile)
    
    def _gaussian_kernel(self, distance: float, bandwidth: float) -> float:
        """Compute Gaussian kernel weight."""
        return np.exp(-(distance**2) / (2 * bandwidth**2))
    
    def _mean_shift_single_seed(
        self,
        seed: np.ndarray,
        X: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """Perform mean shift for a single seed point."""
        current = seed.copy()
        
        for iteration in range(self.max_iter):
            # Compute distances to all points
            distances = np.sqrt(np.sum((X - current)**2, axis=1))
            
            # Compute weights using Gaussian kernel
            weights = np.exp(-(distances**2) / (2 * bandwidth**2))
            
            # Compute weighted mean
            new_center = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Check convergence
            if np.linalg.norm(new_center - current) < 1e-3 * bandwidth:
                break
            
            current = new_center
        
        return current
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MeanShift":
        """Perform mean shift clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : MeanShift
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            bandwidth = self._estimate_bandwidth(X)
        else:
            bandwidth = self.bandwidth
        
        # Use all points as seeds
        seeds = X.copy()
        
        # Perform mean shift for each seed
        centers = []
        
        for seed in seeds:
            center = self._mean_shift_single_seed(seed, X, bandwidth)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Remove duplicate centers (post-processing)
        unique_centers = []
        
        for center in centers:
            # Check if this center is close to any existing unique center
            is_unique = True
            
            for unique_center in unique_centers:
                if np.linalg.norm(center - unique_center) < bandwidth:
                    is_unique = False
                    break
            
            if is_unique:
                unique_centers.append(center)
        
        self.cluster_centers_ = np.array(unique_centers)
        
        # Assign labels based on nearest center
        labels = np.zeros(n_samples, dtype=int)
        
        for i, point in enumerate(X):
            distances = np.sqrt(np.sum((self.cluster_centers_ - point)**2, axis=1))
            
            if self.cluster_all or np.min(distances) < bandwidth:
                labels[i] = np.argmin(distances)
            else:
                labels[i] = -1  # Outlier
        
        self.labels_ = labels
        self.n_iter_ = self.max_iter
        
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(X)
        
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("MeanShift must be fitted before calling predict")
        
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i, point in enumerate(X):
            distances = np.sqrt(np.sum((self.cluster_centers_ - point)**2, axis=1))
            labels[i] = np.argmin(distances)
        
        return labels
