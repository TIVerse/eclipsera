"""Scaling and normalization transformers."""
from typing import Optional, Tuple

import numpy as np

from ..core.base import BaseTransformer
from ..core.validation import check_array


class StandardScaler(BaseTransformer):
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    copy : bool, default=True
        If False, inplace scaling.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    var_ : ndarray of shape (n_features,)
        The variance for each feature in the training set.

    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.preprocessing import StandardScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)
    StandardScaler()
    >>> scaler.transform(X)
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        copy: bool = True,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        y : None
            Ignored.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        X = check_array(X, dtype=np.float64)

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.var_ = np.var(X, axis=0)
            self.scale_ = np.sqrt(self.var_)
            # Avoid division by zero
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.var_ = None
            self.scale_ = None

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        if self.with_mean:
            X -= self.mean_

        if self.with_std:
            X /= self.scale_

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Transformed data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        if self.with_std:
            X *= self.scale_

        if self.with_mean:
            X += self.mean_

        return X


class MinMaxScaler(BaseTransformer):
    """Transform features by scaling to a given range.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        If False, inplace scaling.
    clip : bool, default=False
        If True, clip transformed values to feature_range.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum value in training data.
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum value in training data.
    data_range_ : ndarray of shape (n_features,)
        Per feature range in training data.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        copy: bool = True,
        clip: bool = False,
    ):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute per-feature minimum and maximum.
        y : None
            Ignored.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler.
        """
        X = check_array(X, dtype=np.float64)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # Avoid division by zero
        self.data_range_[self.data_range_ == 0.0] = 1.0

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        X *= self.scale_
        X += self.min_

        if self.clip:
            X = np.clip(X, self.feature_range[0], self.feature_range[1])

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Transformed data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        X -= self.min_
        X /= self.scale_

        return X


class RobustScaler(BaseTransformer):
    """Scale features using statistics that are robust to outliers.

    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to interquartile range.
    quantile_range : tuple (q_min, q_max), default=(25.0, 75.0)
        Quantile range used to calculate scale.
    copy : bool, default=True
        If False, inplace scaling.

    Attributes
    ----------
    center_ : ndarray of shape (n_features,)
        The median value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        The (scaled) interquartile range for each feature.
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RobustScaler":
        """Compute the median and quantiles to be used for scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the median and quantiles.
        y : None
            Ignored.

        Returns
        -------
        self : RobustScaler
            Fitted scaler.
        """
        X = check_array(X, dtype=np.float64)

        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = None

        if self.with_scaling:
            q_min, q_max = self.quantile_range
            q1 = np.percentile(X, q_min, axis=0)
            q3 = np.percentile(X, q_max, axis=0)
            self.scale_ = q3 - q1
            # Avoid division by zero
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = None

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Center and scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        if self.with_centering:
            X -= self.center_

        if self.with_scaling:
            X /= self.scale_

        return X


class MaxAbsScaler(BaseTransformer):
    """Scale features by their maximum absolute value.

    Parameters
    ----------
    copy : bool, default=True
        If False, inplace scaling.

    Attributes
    ----------
    max_abs_ : ndarray of shape (n_features,)
        Per feature maximum absolute value.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    """

    def __init__(self, copy: bool = True):
        self.copy = copy

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MaxAbsScaler":
        """Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature maximum absolute value.
        y : None
            Ignored.

        Returns
        -------
        self : MaxAbsScaler
            Fitted scaler.
        """
        X = check_array(X, dtype=np.float64)

        self.max_abs_ = np.max(np.abs(X), axis=0)
        self.scale_ = self.max_abs_.copy()
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)
        X /= self.scale_
        return X


class Normalizer(BaseTransformer):
    """Normalize samples individually to unit norm.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non-zero sample.
    copy : bool, default=True
        If False, inplace normalization.

    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.preprocessing import Normalizer
    >>> X = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
    >>> normalizer = Normalizer(norm='l2')
    >>> normalizer.fit_transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])
    """

    def __init__(self, norm: str = "l2", copy: bool = True):
        self.norm = norm
        self.copy = copy

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Normalizer":
        """Do nothing and return the estimator unchanged.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : Normalizer
            Fitted normalizer.
        """
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize samples individually to unit norm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to normalize.

        Returns
        -------
        X_normalized : ndarray of shape (n_samples, n_features)
            Normalized array.
        """
        X = check_array(X, dtype=np.float64, copy=self.copy)

        if self.norm == "l1":
            norms = np.abs(X).sum(axis=1)
        elif self.norm == "l2":
            norms = np.sqrt((X ** 2).sum(axis=1))
        elif self.norm == "max":
            norms = np.abs(X).max(axis=1)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        # Avoid division by zero
        norms[norms == 0.0] = 1.0
        X /= norms[:, np.newaxis]

        return X
