"""Recursive Feature Elimination."""
from typing import Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array, check_X_y


class RFE(BaseTransformer):
    """Feature selection by recursive feature elimination (RFE).
    
    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), RFE selects features by recursively
    considering smaller and smaller sets of features. First, the estimator
    is trained on the initial set of features and the importance of each
    feature is obtained. Then, the least important features are pruned from
    current set of features. That procedure is recursively repeated on the
    pruned set until the desired number of features to select is eventually reached.
    
    Parameters
    ----------
    estimator : estimator object
        A supervised learning estimator with a `fit` method that provides
        information about feature importance (e.g., `coef_` or `feature_importances_`).
    n_features_to_select : int or float, default=None
        The number of features to select. If None, half of the features are selected.
        If integer, that many features are selected. If float between 0 and 1,
        it is the fraction of features to select.
    step : int or float, default=1
        If greater than or equal to 1, then step corresponds to the number of
        features to remove at each iteration. If within (0.0, 1.0), then step
        corresponds to the percentage of features to remove at each iteration.
    verbose : int, default=0
        Controls verbosity of output.
        
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.
    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ranking_[i] corresponds to the ranking
        position of the i-th feature. Selected features are assigned rank 1.
        
    Examples
    --------
    >>> from eclipsera.feature_selection import RFE
    >>> from eclipsera.ml import LogisticRegression
    >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> y = [0, 1, 0, 1, 0, 1]
    >>> estimator = LogisticRegression()
    >>> selector = RFE(estimator, n_features_to_select=2)
    >>> selector.fit(X, y)
    RFE(...)
    >>> selector.support_
    array([False,  True,  True])
    """
    
    def __init__(
        self,
        estimator,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
    
    def _get_feature_importances(self, estimator) -> np.ndarray:
        """Get feature importances from estimator."""
        if hasattr(estimator, 'coef_'):
            # Linear models
            coef = estimator.coef_
            if coef.ndim > 1:
                # Multi-class: use absolute sum across classes
                importances = np.abs(coef).sum(axis=0)
            else:
                importances = np.abs(coef)
        elif hasattr(estimator, 'feature_importances_'):
            # Tree-based models
            importances = estimator.feature_importances_
        else:
            raise ValueError(
                "The estimator must have either coef_ or feature_importances_ attribute"
            )
        
        return importances
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFE":
        """Fit the RFE model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : RFE
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        
        # Determine number of features to select
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, float):
            n_features_to_select = int(n_features * self.n_features_to_select)
        else:
            n_features_to_select = self.n_features_to_select
        
        # Initialize support and ranking
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        
        # Recursive elimination
        current_n_features = n_features
        rank = 1
        
        while current_n_features > n_features_to_select:
            # Determine step size
            if isinstance(self.step, float) and 0 < self.step < 1:
                step = max(1, int(current_n_features * self.step))
            else:
                step = int(self.step)
            
            step = min(step, current_n_features - n_features_to_select)
            
            # Get current features
            features = np.where(support)[0]
            X_subset = X[:, features]
            
            # Train estimator
            estimator = self.estimator.__class__(**self.estimator.get_params())
            estimator.fit(X_subset, y)
            
            # Get feature importances
            importances = self._get_feature_importances(estimator)
            
            # Rank features
            ranks = np.argsort(importances)
            
            # Mark least important features for elimination
            for i in range(step):
                support[features[ranks[i]]] = False
                ranking[features[ranks[i]]] = rank + step - i
            
            current_n_features -= step
            rank += step
            
            if self.verbose > 0:
                print(f"Remaining features: {current_n_features}")
        
        # Fit final estimator
        self.estimator_ = self.estimator.__class__(**self.estimator.get_params())
        self.estimator_.fit(X[:, support], y)
        
        self.n_features_ = n_features_to_select
        self.support_ = support
        self.ranking_ = ranking
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce X to selected features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_transformed : ndarray
            The selected features.
        """
        X = check_array(X)
        
        if not hasattr(self, 'support_'):
            raise ValueError("RFE must be fitted before calling transform")
        
        return X[:, self.support_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        X_transformed : ndarray
            The selected features.
        """
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False):
        """Get mask or indices of selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return feature indices. Otherwise, return boolean mask.
            
        Returns
        -------
        support : ndarray
            Mask or indices of selected features.
        """
        if not hasattr(self, 'support_'):
            raise ValueError("RFE must be fitted before calling get_support")
        
        if indices:
            return np.where(self.support_)[0]
        return self.support_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted estimator with selected features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        if not hasattr(self, 'estimator_'):
            raise ValueError("RFE must be fitted before calling predict")
        
        X_transformed = self.transform(X)
        return self.estimator_.predict(X_transformed)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using the fitted estimator with selected features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        score : float
            Score of the estimator.
        """
        if not hasattr(self, 'estimator_'):
            raise ValueError("RFE must be fitted before calling score")
        
        X_transformed = self.transform(X)
        return self.estimator_.score(X_transformed, y)
