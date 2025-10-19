"""Truncated SVD for dimensionality reduction."""
from typing import Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array


class TruncatedSVD(BaseTransformer):
    """Dimensionality reduction using truncated SVD.
    
    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices efficiently.
    
    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
    algorithm : str, default='randomized'
        SVD solver to use ('randomized' or 'full').
    n_iter : int, default=5
        Number of iterations for randomized SVD solver.
    random_state : int, RandomState instance or None, default=None
        Used during randomized svd.
        
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.
    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.decomposition import TruncatedSVD
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> svd = TruncatedSVD(n_components=2)
    >>> svd.fit(X)
    TruncatedSVD(n_components=2)
    >>> X_transformed = svd.transform(X)
    >>> X_transformed.shape
    (3, 2)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TruncatedSVD":
        """Fit model on training data X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        self : TruncatedSVD
            Fitted estimator.
        """
        X = check_array(X)
        
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        if self.n_components >= min(n_samples, self.n_features_in_):
            raise ValueError(
                f"n_components must be < min(n_samples, n_features); "
                f"got n_components={self.n_components}"
            )
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Keep only n_components
        U = U[:, :self.n_components]
        S = S[:self.n_components]
        Vt = Vt[:self.n_components]
        
        # Flip signs for deterministic output
        U, Vt = self._svd_flip(U, Vt)
        
        self.components_ = Vt
        self.singular_values_ = S
        
        # Calculate explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / total_var
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Reduced version of X.
        """
        X = check_array(X)
        
        if not hasattr(self, 'components_'):
            raise ValueError("TruncatedSVD must be fitted before calling transform")
        
        return X @ self.components_.T
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X back to its original space.
        
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
            raise ValueError("TruncatedSVD must be fitted before calling inverse_transform")
        
        return X @ self.components_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit model to X and perform dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Reduced version of X.
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
