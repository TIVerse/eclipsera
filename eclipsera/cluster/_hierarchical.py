"""Hierarchical clustering."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class AgglomerativeClustering(BaseEstimator):
    """Agglomerative Hierarchical Clustering.
    
    Recursively merges pair of clusters of sample data; uses linkage
    distance.
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation.
    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be merged.
        If not None, n_clusters must be None.
        
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import AgglomerativeClustering
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering(n_clusters=2)
    >>> clustering.fit(X)
    AgglomerativeClustering(n_clusters=2)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        distance_threshold: Optional[float] = None,
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between samples."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_cluster_distance(
        self,
        X: np.ndarray,
        cluster1: list,
        cluster2: list,
    ) -> float:
        """Compute distance between two clusters based on linkage."""
        if self.linkage == "single":
            # Minimum distance between any two points
            min_dist = np.inf
            for i in cluster1:
                for j in cluster2:
                    dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                    min_dist = min(min_dist, dist)
            return min_dist
        
        elif self.linkage == "complete":
            # Maximum distance between any two points
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                    max_dist = max(max_dist, dist)
            return max_dist
        
        elif self.linkage == "average":
            # Average distance between all pairs
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                    total_dist += dist
                    count += 1
            return total_dist / count if count > 0 else 0
        
        elif self.linkage == "ward":
            # Ward linkage: minimize within-cluster variance
            # Compute centroids
            centroid1 = np.mean(X[cluster1], axis=0)
            centroid2 = np.mean(X[cluster2], axis=0)
            
            # Merge and compute variance increase
            merged = cluster1 + cluster2
            merged_centroid = np.mean(X[merged], axis=0)
            
            # Variance before merge
            var1 = np.sum((X[cluster1] - centroid1) ** 2)
            var2 = np.sum((X[cluster2] - centroid2) ** 2)
            
            # Variance after merge
            var_merged = np.sum((X[merged] - merged_centroid) ** 2)
            
            # Increase in variance
            return var_merged - (var1 + var2)
        
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AgglomerativeClustering":
        """Fit the hierarchical clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : AgglomerativeClustering
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Initialize: each sample is its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Determine stopping criterion
        if self.distance_threshold is not None:
            target_n_clusters = 1  # Will stop based on distance
        else:
            target_n_clusters = self.n_clusters
        
        # Agglomerative clustering
        while len(clusters) > target_n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_cluster_distance(X, clusters[i], clusters[j])
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Check distance threshold
            if self.distance_threshold is not None and min_dist > self.distance_threshold:
                break
            
            # Merge the two closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = cluster_id
        
        self.labels_ = labels
        self.n_clusters_ = len(clusters)
        
        return self
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and return cluster labels.
        
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
