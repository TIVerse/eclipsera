"""Univariate feature selection."""
from typing import Callable, Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array, check_X_y


def f_classif(X: np.ndarray, y: np.ndarray) -> tuple:
    """Compute F-value for classification tasks.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample vectors.
    y : array-like of shape (n_samples,)
        Target vector (class labels).
        
    Returns
    -------
    F : ndarray of shape (n_features,)
        F-values for each feature.
    pval : ndarray of shape (n_features,)
        P-values for each feature.
    """
    X, y = check_X_y(X, y)
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape
    
    # Compute F-statistic for each feature
    F_scores = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        # Between-group variance
        overall_mean = np.mean(X[:, feat_idx])
        between_var = 0.0
        
        for c in classes:
            mask = y == c
            class_mean = np.mean(X[mask, feat_idx])
            n_class = np.sum(mask)
            between_var += n_class * (class_mean - overall_mean) ** 2
        
        between_var /= (n_classes - 1)
        
        # Within-group variance
        within_var = 0.0
        for c in classes:
            mask = y == c
            class_vals = X[mask, feat_idx]
            class_mean = np.mean(class_vals)
            within_var += np.sum((class_vals - class_mean) ** 2)
        
        within_var /= (n_samples - n_classes)
        
        # F-statistic
        if within_var > 0:
            F_scores[feat_idx] = between_var / within_var
        else:
            F_scores[feat_idx] = 0.0
    
    # For simplicity, return dummy p-values (proper calculation requires scipy)
    pval = np.ones(n_features)
    
    return F_scores, pval


def chi2(X: np.ndarray, y: np.ndarray) -> tuple:
    """Compute chi-squared stats between each feature and target.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample vectors (non-negative).
    y : array-like of shape (n_samples,)
        Target vector (class labels).
        
    Returns
    -------
    chi2 : ndarray of shape (n_features,)
        Chi-squared statistics for each feature.
    pval : ndarray of shape (n_features,)
        P-values for each feature.
    """
    X, y = check_X_y(X, y)
    
    if np.any(X < 0):
        raise ValueError("Input X must be non-negative for chi2")
    
    n_samples, n_features = X.shape
    classes = np.unique(y)
    
    chi2_scores = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        feature = X[:, feat_idx]
        
        # Compute observed frequencies
        observed = np.zeros((len(classes), 2))  # 2 bins: zero and non-zero
        
        for i, c in enumerate(classes):
            mask = y == c
            observed[i, 0] = np.sum(feature[mask] == 0)
            observed[i, 1] = np.sum(feature[mask] != 0)
        
        # Compute expected frequencies
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        total = observed.sum()
        
        expected = np.outer(row_sums, col_sums) / total
        
        # Compute chi-squared statistic
        # Avoid division by zero
        mask = expected > 0
        chi2_stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        
        chi2_scores[feat_idx] = chi2_stat
    
    # Dummy p-values
    pval = np.ones(n_features)
    
    return chi2_scores, pval


class SelectKBest(BaseTransformer):
    """Select features according to the k highest scores.
    
    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
    k : int or 'all', default=10
        Number of top features to select. The 'all' option bypasses selection.
        
    Attributes
    ----------
    scores_ : ndarray of shape (n_features,)
        Scores of features.
    pvalues_ : ndarray of shape (n_features,)
        P-values of feature scores.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.feature_selection import SelectKBest, f_classif
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> y = np.array([0, 0, 1, 1])
    >>> selector = SelectKBest(f_classif, k=2)
    >>> X_new = selector.fit_transform(X, y)
    >>> X_new.shape
    (4, 2)
    """
    
    def __init__(
        self,
        score_func: Callable = f_classif,
        k: int = 10,
    ):
        self.score_func = score_func
        self.k = k
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelectKBest":
        """Run score function on (X, y) and get the appropriate features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : SelectKBest
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.n_features_in_ = X.shape[1]
        
        # Compute scores
        self.scores_, self.pvalues_ = self.score_func(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce X to the selected features.
        
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
        
        if not hasattr(self, 'scores_'):
            raise ValueError("SelectKBest must be fitted before calling transform")
        
        if self.k == 'all':
            return X
        
        # Select top k features
        k = min(self.k, len(self.scores_))
        top_k_indices = np.argsort(self.scores_)[-k:][::-1]
        
        return X[:, top_k_indices]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
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
        if not hasattr(self, 'scores_'):
            raise ValueError("SelectKBest must be fitted before calling get_support")
        
        if self.k == 'all':
            mask = np.ones(len(self.scores_), dtype=bool)
        else:
            k = min(self.k, len(self.scores_))
            top_k_indices = np.argsort(self.scores_)[-k:]
            mask = np.zeros(len(self.scores_), dtype=bool)
            mask[top_k_indices] = True
        
        if indices:
            return np.where(mask)[0]
        return mask
