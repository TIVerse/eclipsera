"""Encoding categorical features."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseTransformer
from ..core.exceptions import NotFittedError
from ..core.validation import check_array


class LabelEncoder(BaseTransformer):
    """Encode target labels with value between 0 and n_classes-1.
    
    This transformer should be used to encode target values, i.e. y,
    and not the input X.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.
        
    Examples
    --------
    >>> from eclipsera.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> le.fit(['paris', 'tokyo', 'paris', 'amsterdam'])
    LabelEncoder()
    >>> le.classes_
    array(['amsterdam', 'paris', 'tokyo'], dtype='<U9')
    >>> le.transform(['tokyo', 'paris', 'amsterdam'])
    array([2, 1, 0])
    >>> le.inverse_transform([2, 1, 0])
    array(['tokyo', 'paris', 'amsterdam'], dtype='<U9')
    """
    
    def fit(self, y: np.ndarray) -> "LabelEncoder":
        """Fit label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : LabelEncoder
            Fitted encoder.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to normalized encoding.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        y_encoded : ndarray of shape (n_samples,)
            Encoded labels.
        """
        y = np.asarray(y)
        
        if not hasattr(self, 'classes_'):
            raise NotFittedError("This LabelEncoder instance is not fitted yet.")
        
        # Create mapping
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        # Transform
        return np.array([class_to_idx.get(val, -1) for val in y])
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels back to original encoding.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Encoded labels.
            
        Returns
        -------
        y_decoded : ndarray of shape (n_samples,)
            Original labels.
        """
        y = np.asarray(y)
        
        if not hasattr(self, 'classes_'):
            raise NotFittedError("This LabelEncoder instance is not fitted yet.")
        
        return self.classes_[y]
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit label encoder and return encoded labels.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        y_encoded : ndarray of shape (n_samples,)
            Encoded labels.
        """
        return self.fit(y).transform(y)


class OneHotEncoder(BaseTransformer):
    """Encode categorical features as a one-hot numeric array.
    
    Parameters
    ----------
    sparse : bool, default=False
        Will return sparse matrix if set True (not implemented yet).
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical
        feature is present during transform.
        
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> from eclipsera.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder()
    >>> enc.transform([['Female', 1], ['Male', 4]])
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 1.]])
    """
    
    def __init__(
        self,
        sparse: bool = False,
        handle_unknown: Literal["error", "ignore"] = "error",
    ):
        self.sparse = sparse
        self.handle_unknown = handle_unknown
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneHotEncoder":
        """Fit OneHotEncoder to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored.
            
        Returns
        -------
        self : OneHotEncoder
            Fitted encoder.
        """
        X = check_array(X, dtype=None, force_all_finite=False)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        
        for i in range(self.n_features_in_):
            categories = np.unique(X[:, i])
            self.categories_.append(categories)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using one-hot encoding.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
            
        Returns
        -------
        X_encoded : ndarray
            Transformed input.
        """
        X = check_array(X, dtype=None, force_all_finite=False)
        
        if not hasattr(self, 'categories_'):
            raise NotFittedError("This OneHotEncoder instance is not fitted yet.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder "
                f"is expecting {self.n_features_in_} features as input."
            )
        
        encoded_features = []
        
        for i in range(self.n_features_in_):
            feature_categories = self.categories_[i]
            n_categories = len(feature_categories)
            
            # Create mapping
            cat_to_idx = {cat: idx for idx, cat in enumerate(feature_categories)}
            
            # Encode this feature
            encoded = np.zeros((X.shape[0], n_categories))
            
            for j, val in enumerate(X[:, i]):
                if val in cat_to_idx:
                    encoded[j, cat_to_idx[val]] = 1
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{val}' in feature {i}")
                # If 'ignore', leave as zeros
            
            encoded_features.append(encoded)
        
        return np.hstack(encoded_features)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit OneHotEncoder to X, then transform X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
        y : None
            Ignored.
            
        Returns
        -------
        X_encoded : ndarray
            Transformed input.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert back from one-hot encoding.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The one-hot encoded data.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        if not hasattr(self, 'categories_'):
            raise NotFittedError("This OneHotEncoder instance is not fitted yet.")
        
        X = check_array(X)
        
        result = []
        start_idx = 0
        
        for i in range(self.n_features_in_):
            n_categories = len(self.categories_[i])
            end_idx = start_idx + n_categories
            
            # Get the column with the 1
            feature_encoded = X[:, start_idx:end_idx]
            indices = np.argmax(feature_encoded, axis=1)
            
            # Map back to categories
            decoded = self.categories_[i][indices]
            result.append(decoded.reshape(-1, 1))
            
            start_idx = end_idx
        
        return np.hstack(result)


class OrdinalEncoder(BaseTransformer):
    """Encode categorical features as an integer array.
    
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    
    Parameters
    ----------
    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error', raise an error if an unknown category is seen.
        When set to 'use_encoded_value', use the value given for the parameter
        unknown_value.
    unknown_value : int, default=None
        When handle_unknown is set to 'use_encoded_value', this parameter is
        required.
        
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature.
        
    Examples
    --------
    >>> from eclipsera.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])
    """
    
    def __init__(
        self,
        handle_unknown: Literal["error", "use_encoded_value"] = "error",
        unknown_value: Optional[int] = None,
    ):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OrdinalEncoder":
        """Fit OrdinalEncoder to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored.
            
        Returns
        -------
        self : OrdinalEncoder
            Fitted encoder.
        """
        X = check_array(X, dtype=None, force_all_finite=False)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        
        for i in range(self.n_features_in_):
            categories = np.unique(X[:, i])
            self.categories_.append(categories)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to ordinal codes.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
            
        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X = check_array(X, dtype=None, force_all_finite=False)
        
        if not hasattr(self, 'categories_'):
            raise NotFittedError("This OrdinalEncoder instance is not fitted yet.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OrdinalEncoder "
                f"is expecting {self.n_features_in_} features as input."
            )
        
        X_encoded = np.zeros_like(X, dtype=float)
        
        for i in range(self.n_features_in_):
            categories = self.categories_[i]
            cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            
            for j, val in enumerate(X[:, i]):
                if val in cat_to_idx:
                    X_encoded[j, i] = cat_to_idx[val]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{val}' in feature {i}")
                elif self.handle_unknown == 'use_encoded_value':
                    if self.unknown_value is None:
                        raise ValueError(
                            "unknown_value must be set when handle_unknown='use_encoded_value'"
                        )
                    X_encoded[j, i] = self.unknown_value
        
        return X_encoded
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert back from ordinal codes.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The ordinal encoded data.
            
        Returns
        -------
        X_original : ndarray
            Inverse transformed array.
        """
        if not hasattr(self, 'categories_'):
            raise NotFittedError("This OrdinalEncoder instance is not fitted yet.")
        
        X = check_array(X)
        
        X_decoded = np.empty_like(X, dtype=object)
        
        for i in range(self.n_features_in_):
            categories = self.categories_[i]
            
            for j, val in enumerate(X[:, i]):
                idx = int(val)
                if 0 <= idx < len(categories):
                    X_decoded[j, i] = categories[idx]
                else:
                    X_decoded[j, i] = None  # Unknown value
        
        return X_decoded
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit OrdinalEncoder to X, then transform X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
        y : None
            Ignored.
            
        Returns
        -------
        X_encoded : ndarray
            Transformed input.
        """
        return self.fit(X, y).transform(X)
