"""Principal Component Analysis."""
from typing import Optional, Union

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array


class PCA(BaseTransformer):
    """Principal Component Analysis (PCA).
    
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.
    
    Parameters
    ----------
    n_components : int, float or None, default=None
        Number of components to keep.
        If None, all components are kept.
        If 0 < n_components < 1, select the number of components such that
        the explained variance is greater than the percentage specified.
    whiten : bool, default=False
        When True, the components_ vectors are divided by n_samples times
        singular values to ensure uncorrelated outputs with unit variance.
    svd_solver : str, default='auto'
        Solver to use ('auto', 'full', 'randomized').
    random_state : int, RandomState instance or None, default=None
        Used when svd_solver == 'randomized'.
        
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray of shape (n_components,)
        Amount of variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component.
    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(n_components=2)
    >>> print(pca.explained_variance_ratio_)
    [0.99244289 0.00755711]
    >>> X_transformed = pca.transform(X)
    """
    
    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCA":
        """Fit the model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Flip signs for deterministic output
        U, Vt = self._svd_flip(U, Vt)
        
        # Get components
        components = Vt
        
        # Explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance
        
        # Determine number of components to keep
        if self.n_components is None:
            n_components = min(n_samples, self.n_features_in_)
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Select components that explain desired variance
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
        else:
            n_components = min(int(self.n_components), len(S))
        
        # Store results
        self.components_ = components[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = S[:n_components]
        self.n_components_ = n_components
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = check_array(X)
        
        if not hasattr(self, 'components_'):
            raise ValueError("PCA must be fitted before calling transform")
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto components
        X_transformed = X_centered @ self.components_.T
        
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Transformed data.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        X = check_array(X)
        
        if not hasattr(self, 'components_'):
            raise ValueError("PCA must be fitted before calling inverse_transform")
        
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        
        return X @ self.components_ + self.mean_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _svd_flip(self, U: np.ndarray, V: np.ndarray) -> tuple:
        """Sign correction to ensure deterministic output."""
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]
        return U, V
