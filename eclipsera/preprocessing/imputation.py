"""Imputation of missing values."""
from typing import Literal, Optional, Union

import numpy as np

from ..core.base import BaseTransformer
from ..core.exceptions import NotFittedError
from ..core.validation import check_array


class SimpleImputer(BaseTransformer):
    """Imputation transformer for completing missing values.
    
    Parameters
    ----------
    missing_values : int, float or None, default=np.nan
        The placeholder for the missing values.
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        The imputation strategy.
    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        
    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.preprocessing import SimpleImputer
    >>> imp = SimpleImputer(strategy='mean')
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> imp.fit(X)
    SimpleImputer()
    >>> X_imputed = imp.transform(X)
    >>> X_imputed
    array([[1., 2.],
           [4., 3.],
           [7., 6.]])
    """
    
    def __init__(
        self,
        missing_values: Union[int, float, None] = np.nan,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
        fill_value: Optional[Union[str, float]] = None,
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
    
    def _is_missing(self, X: np.ndarray) -> np.ndarray:
        """Identify missing values in X."""
        if self.missing_values is None or (
            isinstance(self.missing_values, float) and np.isnan(self.missing_values)
        ):
            return np.isnan(X)
        else:
            return X == self.missing_values
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SimpleImputer":
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.
            
        Returns
        -------
        self : SimpleImputer
            Fitted imputer.
        """
        X = check_array(X, dtype=float, force_all_finite=False)
        
        self.n_features_in_ = X.shape[1]
        
        # Compute statistics for each feature
        self.statistics_ = np.zeros(self.n_features_in_)
        
        for i in range(self.n_features_in_):
            feature = X[:, i]
            mask = ~self._is_missing(feature.reshape(-1, 1)).ravel()
            
            if self.strategy == "mean":
                self.statistics_[i] = np.mean(feature[mask]) if mask.any() else 0
            elif self.strategy == "median":
                self.statistics_[i] = np.median(feature[mask]) if mask.any() else 0
            elif self.strategy == "most_frequent":
                if mask.any():
                    values, counts = np.unique(feature[mask], return_counts=True)
                    self.statistics_[i] = values[np.argmax(counts)]
                else:
                    self.statistics_[i] = 0
            elif self.strategy == "constant":
                if self.fill_value is None:
                    raise ValueError("fill_value must be set when strategy='constant'")
                self.statistics_[i] = self.fill_value
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute all missing values in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed input data.
        """
        X = check_array(X, dtype=float, force_all_finite=False, copy=True)
        
        if not hasattr(self, 'statistics_'):
            raise NotFittedError("This SimpleImputer instance is not fitted yet.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SimpleImputer "
                f"is expecting {self.n_features_in_} features as input."
            )
        
        # Impute each feature
        for i in range(self.n_features_in_):
            mask = self._is_missing(X[:, i].reshape(-1, 1)).ravel()
            X[mask, i] = self.statistics_[i]
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed input data.
        """
        return self.fit(X, y).transform(X)


class KNNImputer(BaseTransformer):
    """Imputation using k-Nearest Neighbors.
    
    Each missing value is imputed using the mean value from n_neighbors
    nearest neighbors found in the training set.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    missing_values : int, float or None, default=np.nan
        The placeholder for the missing values.
        
    Attributes
    ----------
    X_ : ndarray
        The training data used for imputation.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.preprocessing import KNNImputer
    >>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        missing_values: Union[int, float, None] = np.nan,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.missing_values = missing_values
    
    def _is_missing(self, X: np.ndarray) -> np.ndarray:
        """Identify missing values in X."""
        if self.missing_values is None or (
            isinstance(self.missing_values, float) and np.isnan(self.missing_values)
        ):
            return np.isnan(X)
        else:
            return X == self.missing_values
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KNNImputer":
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.
            
        Returns
        -------
        self : KNNImputer
            Fitted imputer.
        """
        X = check_array(X, dtype=float, force_all_finite=False)
        
        self.n_features_in_ = X.shape[1]
        self.X_ = X.copy()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute all missing values in X using KNN.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed input data.
        """
        X = check_array(X, dtype=float, force_all_finite=False, copy=True)
        
        if not hasattr(self, 'X_'):
            raise NotFittedError("This KNNImputer instance is not fitted yet.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNNImputer "
                f"is expecting {self.n_features_in_} features as input."
            )
        
        # For each sample with missing values
        for i in range(X.shape[0]):
            sample = X[i]
            missing_mask = self._is_missing(sample.reshape(1, -1)).ravel()
            
            if not missing_mask.any():
                continue  # No missing values in this sample
            
            # Find k nearest neighbors based on non-missing features
            non_missing_mask = ~missing_mask
            
            # Compute distances to all training samples
            distances = []
            valid_indices = []
            
            for j in range(self.X_.shape[0]):
                train_sample = self.X_[j]
                
                # Only use features that are non-missing in both samples
                common_mask = non_missing_mask & ~self._is_missing(train_sample.reshape(1, -1)).ravel()
                
                if common_mask.any():
                    dist = np.sqrt(np.sum((sample[common_mask] - train_sample[common_mask]) ** 2))
                    distances.append(dist)
                    valid_indices.append(j)
            
            if not valid_indices:
                continue  # No valid neighbors
            
            # Get k nearest neighbors
            distances = np.array(distances)
            valid_indices = np.array(valid_indices)
            
            k = min(self.n_neighbors, len(distances))
            nearest_indices = valid_indices[np.argsort(distances)[:k]]
            nearest_distances = distances[np.argsort(distances)[:k]]
            
            # Impute missing features
            for feat_idx in np.where(missing_mask)[0]:
                # Get values from neighbors for this feature
                neighbor_values = []
                neighbor_weights = []
                
                for neighbor_idx, dist in zip(nearest_indices, nearest_distances):
                    neighbor_value = self.X_[neighbor_idx, feat_idx]
                    if not self._is_missing(np.array([[neighbor_value]]))[0, 0]:
                        neighbor_values.append(neighbor_value)
                        if self.weights == "distance":
                            # Avoid division by zero
                            weight = 1 / (dist + 1e-10)
                        else:
                            weight = 1.0
                        neighbor_weights.append(weight)
                
                if neighbor_values:
                    neighbor_values = np.array(neighbor_values)
                    neighbor_weights = np.array(neighbor_weights)
                    
                    # Weighted average
                    X[i, feat_idx] = np.sum(neighbor_values * neighbor_weights) / np.sum(neighbor_weights)
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed input data.
        """
        return self.fit(X, y).transform(X)
