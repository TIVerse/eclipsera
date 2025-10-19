"""K-Nearest Neighbors algorithms for classification and regression."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseClassifier, BaseRegressor
from ..core.exceptions import NotFittedError
from ..core.validation import check_array, check_X_y


class KNeighborsClassifier(BaseClassifier):
    """K-Nearest Neighbors Classifier.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    algorithm : {'auto', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    metric : str, default='euclidean'
        Distance metric to use.
    p : int, default=2
        Power parameter for the Minkowski metric.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    y_ : ndarray of shape (n_samples,)
        Training labels.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.neighbors import KNeighborsClassifier
    >>> X = np.array([[0], [1], [2], [3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KNeighborsClassifier(n_neighbors=3)
    >>> clf.fit(X, y)
    KNeighborsClassifier(n_neighbors=3)
    >>> clf.predict([[1.1]])
    array([0])
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        algorithm: str = "auto",
        metric: str = "euclidean",
        p: int = 2,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        """Fit the k-nearest neighbors classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : KNeighborsClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Store training data
        self.X_ = X.copy()
        self.y_ = y.copy()
        
        return self
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances between X and training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Query samples.
            
        Returns
        -------
        distances : ndarray of shape (n_samples, n_training_samples)
            Distances to training samples.
        """
        if self.metric == "euclidean":
            # Euclidean distance
            distances = np.sqrt(
                np.sum((X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** 2, axis=2)
            )
        elif self.metric == "manhattan":
            # Manhattan distance
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]), axis=2
            )
        elif self.metric == "minkowski":
            # Minkowski distance
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** self.p,
                axis=2
            ) ** (1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def _get_neighbors(self, X: np.ndarray) -> tuple:
        """Get k-nearest neighbors for each sample in X.
        
        Returns
        -------
        neighbor_indices : ndarray
            Indices of k-nearest neighbors.
        neighbor_distances : ndarray
            Distances to k-nearest neighbors.
        """
        distances = self._compute_distances(X)
        
        # Get indices of k nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Get corresponding distances
        neighbor_distances = np.take_along_axis(
            distances, neighbor_indices, axis=1
        )
        
        return neighbor_indices, neighbor_distances
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X = check_array(X)
        
        if not hasattr(self, 'X_'):
            raise NotFittedError("This KNeighborsClassifier instance is not fitted yet.")
        
        neighbor_indices, neighbor_distances = self._get_neighbors(X)
        
        # Get labels of neighbors
        neighbor_labels = self.y_[neighbor_indices]
        
        # Calculate weights
        if self.weights == "uniform":
            weights = np.ones_like(neighbor_distances)
        elif self.weights == "distance":
            # Avoid division by zero
            weights = 1 / (neighbor_distances + 1e-10)
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
        
        # Calculate class probabilities
        probas = np.zeros((X.shape[0], self.n_classes_))
        
        for i in range(X.shape[0]):
            for j, class_label in enumerate(self.classes_):
                mask = neighbor_labels[i] == class_label
                probas[i, j] = np.sum(weights[i][mask])
        
        # Normalize
        probas /= probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class KNeighborsRegressor(BaseRegressor):
    """K-Nearest Neighbors Regressor.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    algorithm : {'auto', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    metric : str, default='euclidean'
        Distance metric to use.
    p : int, default=2
        Power parameter for the Minkowski metric.
        
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    y_ : ndarray of shape (n_samples,)
        Training target values.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        algorithm: str = "auto",
        metric: str = "euclidean",
        p: int = 2,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsRegressor":
        """Fit the k-nearest neighbors regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
            
        Returns
        -------
        self : KNeighborsRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        self.n_features_in_ = X.shape[1]
        
        # Store training data
        self.X_ = X.copy()
        self.y_ = y.copy()
        
        return self
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances between X and training data."""
        if self.metric == "euclidean":
            distances = np.sqrt(
                np.sum((X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** 2, axis=2)
            )
        elif self.metric == "manhattan":
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]), axis=2
            )
        elif self.metric == "minkowski":
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** self.p,
                axis=2
            ) ** (1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def _get_neighbors(self, X: np.ndarray) -> tuple:
        """Get k-nearest neighbors for each sample in X."""
        distances = self._compute_distances(X)
        
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbor_distances = np.take_along_axis(
            distances, neighbor_indices, axis=1
        )
        
        return neighbor_indices, neighbor_distances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target for the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'X_'):
            raise NotFittedError("This KNeighborsRegressor instance is not fitted yet.")
        
        neighbor_indices, neighbor_distances = self._get_neighbors(X)
        
        # Get target values of neighbors
        neighbor_targets = self.y_[neighbor_indices]
        
        # Calculate weights
        if self.weights == "uniform":
            weights = np.ones_like(neighbor_distances)
        elif self.weights == "distance":
            weights = 1 / (neighbor_distances + 1e-10)
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
        
        # Weighted average of neighbor targets
        if self.y_.ndim == 1:
            # Single output
            predictions = np.sum(
                neighbor_targets * weights, axis=1
            ) / np.sum(weights, axis=1)
        else:
            # Multiple outputs
            weights_expanded = weights[:, :, np.newaxis]
            predictions = np.sum(
                neighbor_targets * weights_expanded, axis=1
            ) / np.sum(weights, axis=1, keepdims=True)
        
        return predictions
