"""K-Means clustering."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.utils import check_random_state
from ..core.validation import check_array


class KMeans(BaseEstimator):
    """K-Means clustering.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm.
    tol : float, default=1e-4
        Relative tolerance to declare convergence.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    verbose : int, default=0
        Verbosity mode.
        
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0)
    >>> kmeans.fit(X)
    KMeans(n_clusters=2, random_state=0)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([0, 1])
    >>> kmeans.cluster_centers_
    array([[1., 2.],
           [10., 2.]])
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
    
    def _init_centroids(self, X: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """Initialize centroids."""
        n_samples = X.shape[0]
        
        if self.init == "k-means++":
            # K-means++ initialization
            centroids = []
            
            # Choose first centroid randomly
            first_idx = random_state.randint(n_samples)
            centroids.append(X[first_idx])
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = np.min([
                    np.sum((X - centroid) ** 2, axis=1)
                    for centroid in centroids
                ], axis=0)
                
                # Choose next centroid with probability proportional to distance squared
                probabilities = distances / distances.sum()
                cumulative_probabilities = np.cumsum(probabilities)
                r = random_state.rand()
                
                for idx, cum_prob in enumerate(cumulative_probabilities):
                    if r < cum_prob:
                        centroids.append(X[idx])
                        break
            
            return np.array(centroids)
        
        elif self.init == "random":
            # Random initialization
            indices = random_state.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()
        
        else:
            raise ValueError(f"init should be either 'k-means++' or 'random', got {self.init}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign samples to nearest centroid."""
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned samples."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute sum of squared distances to closest cluster center."""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                inertia += np.sum((X[mask] - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KMeans":
        """Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        random_state = check_random_state(self.random_state)
        
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        # Run k-means n_init times and keep best result
        for init_idx in range(self.n_init):
            # Initialize centroids
            centroids = self._init_centroids(X, random_state)
            
            # Run k-means iterations
            for iteration in range(self.max_iter):
                # Assign clusters
                labels = self._assign_clusters(X, centroids)
                
                # Update centroids
                new_centroids = self._update_centroids(X, labels)
                
                # Check convergence
                centroid_shift = np.sum((new_centroids - centroids) ** 2)
                
                centroids = new_centroids
                
                if centroid_shift < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)
            
            if self.verbose:
                print(f"Initialization {init_idx + 1}/{self.n_init}: inertia = {inertia:.4f}")
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
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
            raise ValueError("KMeans must be fitted before calling predict")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        X = check_array(X)
        
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("KMeans must be fitted before calling transform")
        
        # Return distances to each cluster center
        distances = np.sqrt(((X[:, np.newaxis] - self.cluster_centers_) ** 2).sum(axis=2))
        return distances
    
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Opposite of the value of X on the K-means objective.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        score : float
            Negative inertia.
        """
        X = check_array(X)
        
        labels = self.predict(X)
        inertia = self._compute_inertia(X, labels, self.cluster_centers_)
        
        return -inertia


class MiniBatchKMeans(KMeans):
    """Mini-Batch K-Means clustering.
    
    Alternative to KMeans that uses mini-batches to reduce computation time,
    while still attempting to optimize the same objective function.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    batch_size : int, default=100
        Size of the mini batches.
    max_iter : int, default=100
        Maximum number of iterations over the complete dataset.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    verbose : int, default=0
        Verbosity mode.
        
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.cluster import MiniBatchKMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> mbk = MiniBatchKMeans(n_clusters=2, batch_size=3, random_state=0)
    >>> mbk.fit(X)
    MiniBatchKMeans(batch_size=3, n_clusters=2, random_state=0)
    >>> mbk.labels_
    array([0, 0, 0, 1, 1, 1])
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 100,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,  # Mini-batch only does one initialization
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
        )
        self.batch_size = batch_size
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MiniBatchKMeans":
        """Compute k-means clustering using mini-batches.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : MiniBatchKMeans
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)
        
        # Initialize centroids
        self.cluster_centers_ = self._init_centroids(X, random_state)
        
        # Counts for each cluster (for weighted average)
        counts = np.zeros(self.n_clusters)
        
        # Mini-batch iterations
        for iteration in range(self.max_iter):
            # Sample mini-batch
            batch_indices = random_state.choice(
                n_samples,
                size=min(self.batch_size, n_samples),
                replace=False
            )
            X_batch = X[batch_indices]
            
            # Assign clusters for batch
            labels_batch = self._assign_clusters(X_batch, self.cluster_centers_)
            
            # Update centroids incrementally
            for k in range(self.n_clusters):
                mask = labels_batch == k
                if mask.any():
                    # Weighted average update
                    counts[k] += mask.sum()
                    eta = 1.0 / counts[k]  # Learning rate
                    self.cluster_centers_[k] = (
                        (1 - eta) * self.cluster_centers_[k] +
                        eta * X_batch[mask].mean(axis=0)
                    )
        
        # Final assignment of all samples
        self.labels_ = self._assign_clusters(X, self.cluster_centers_)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.cluster_centers_)
        self.n_iter_ = self.max_iter
        
        return self
