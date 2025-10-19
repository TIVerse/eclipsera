"""Variance-based feature selection."""
from typing import Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array


class VarianceThreshold(BaseTransformer):
    """Feature selector that removes low-variance features.
    
    Features with a variance lower than the threshold will be removed.
    By default, it removes all zero-variance features.
    
    Parameters
    ----------
    threshold : float, default=0.0
        Features with a variance lower than this threshold will be removed.
        
    Attributes
    ----------
    variances_ : ndarray of shape (n_features,)
        Variances of individual features.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.feature_selection import VarianceThreshold
    >>> X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    >>> selector = VarianceThreshold(threshold=0.1)
    >>> selector.fit_transform(X)
    array([[2, 0],
           [1, 4],
           [1, 1]])
    """
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VarianceThreshold":
        """Learn empirical variances from X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute variances.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : VarianceThreshold
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        
        # Compute variance for each feature
        self.variances_ = np.var(X, axis=0)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove features with low variance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray
            Array with selected features.
        """
        X = check_array(X)
        
        if not hasattr(self, 'variances_'):
            raise ValueError("VarianceThreshold must be fitted before calling transform")
        
        # Select features with variance above threshold
        mask = self.variances_ > self.threshold
        
        if not mask.any():
            raise ValueError("No features have variance above threshold")
        
        return X[:, mask]
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to fit and transform.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        X_transformed : ndarray
            Array with selected features.
        """
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False):
        """Get a mask, or integer index, of the features selected.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return feature indices. Otherwise, return boolean mask.
            
        Returns
        -------
        support : ndarray
            Mask or indices of selected features.
        """
        if not hasattr(self, 'variances_'):
            raise ValueError("VarianceThreshold must be fitted before calling get_support")
        
        mask = self.variances_ > self.threshold
        
        if indices:
            return np.where(mask)[0]
        return mask
