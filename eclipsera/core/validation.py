"""Input validation utilities for Eclipsera.

This module provides functions for validating and preprocessing input data,
ensuring consistency and correctness across the framework.
"""
import numbers
import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.random import RandomState

from .exceptions import (
    DataConversionWarning,
    DataDimensionalityError,
    FeatureNamesError,
    ValidationError,
)


def check_array(
    array: Any,
    accept_sparse: Union[bool, str, list[str]] = False,
    accept_large_sparse: bool = True,
    dtype: Union[str, type, list, None] = "numeric",
    order: Optional[str] = None,
    copy: bool = False,
    force_all_finite: Union[bool, str] = True,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    estimator: Optional[Any] = None,
) -> np.ndarray:
    """Input validation on an array, list, sparse matrix or similar.

    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : bool, str or list of str, default=False
        String[s] representing allowed sparse matrix formats ('csr', 'csc', etc.).
        If input is sparse but not in allowed format, it will be converted.
    accept_large_sparse : bool, default=True
        If False, large sparse matrices will be rejected.
    dtype : str, type, list of type or None, default='numeric'
        Data type of result. If None, dtype of input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : bool, default=False
        Whether a forced copy will be triggered.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array.
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.
    ensure_min_samples : int, default=1
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.
    estimator : str or estimator instance, default=None
        If passed, include name in error messages.

    Returns
    -------
    array_converted : ndarray or sparse matrix
        The converted and validated array.

    Raises
    ------
    ValueError
        If the array does not meet the specified criteria.
    """
    # Convert to array if necessary
    if sp.issparse(array):
        if not accept_sparse:
            raise TypeError(
                f"A sparse matrix was passed, but dense data is required. "
                f"Use X.toarray() to convert to a dense numpy array."
            )
        # Validate sparse format
        if isinstance(accept_sparse, (list, tuple)):
            if array.format not in accept_sparse:
                array = array.asformat(accept_sparse[0])
        elif isinstance(accept_sparse, str):
            if array.format != accept_sparse:
                array = array.asformat(accept_sparse)
        
        # Check for finite values in sparse matrix
        if force_all_finite:
            if force_all_finite == "allow-nan":
                if not np.isfinite(array.data).all():
                    if not np.isnan(array.data).any():
                        raise ValueError("Input contains infinity.")
            else:
                if not np.isfinite(array.data).all():
                    raise ValueError("Input contains NaN or infinity.")
        
        return array
    
    # Handle pandas DataFrames/Series
    if isinstance(array, (pd.DataFrame, pd.Series)):
        array = array.values
    
    # Convert to numpy array
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=dtype if dtype != "numeric" else None)
    else:
        if copy:
            array = np.array(array, copy=True, order=order)
        elif order is not None and array.flags["C_CONTIGUOUS"] != (order == "C"):
            array = np.array(array, copy=False, order=order)
    
    # Handle dtype conversion
    if dtype == "numeric":
        if array.dtype.kind == "O":
            # Try to convert object array to numeric
            try:
                array = array.astype(np.float64)
            except (ValueError, TypeError):
                raise ValueError(
                    "Unable to convert array of object dtype to numeric. "
                    "Ensure all elements are numeric."
                )
    elif dtype is not None:
        if isinstance(dtype, (list, tuple)):
            if array.dtype not in dtype:
                array = array.astype(dtype[0])
        else:
            if array.dtype != dtype:
                array = array.astype(dtype)
    
    # Check for finite values
    if force_all_finite:
        _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
    
    # Check dimensions
    if ensure_2d and array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {array.ndim}D array instead. "
            f"Reshape your data using array.reshape(-1, 1) if your data has a single feature "
            f"or array.reshape(1, -1) if it contains a single sample."
        )
    
    if not allow_nd and array.ndim > 2:
        raise ValueError(f"Found array with dim {array.ndim}. Expected <= 2.")
    
    # Check minimum samples and features
    if array.ndim >= 1 and array.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {array.shape[0]} sample(s) while a minimum of "
            f"{ensure_min_samples} is required."
        )
    
    if array.ndim >= 2 and array.shape[1] < ensure_min_features:
        raise ValueError(
            f"Found array with {array.shape[1]} feature(s) while a minimum of "
            f"{ensure_min_features} is required."
        )
    
    return array


def check_X_y(
    X: Any,
    y: Any,
    accept_sparse: Union[bool, str, list[str]] = False,
    accept_large_sparse: bool = True,
    dtype: Union[str, type, list, None] = "numeric",
    order: Optional[str] = None,
    copy: bool = False,
    force_all_finite: Union[bool, str] = True,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    multi_output: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    y_numeric: bool = False,
    estimator: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D (unless
    multi_output=True). Standard input checks are also applied to y.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.
    y : array-like
        Labels.
    accept_sparse : bool, str or list of str, default=False
        String[s] representing allowed sparse matrix formats.
    accept_large_sparse : bool, default=True
        If False, large sparse matrices will be rejected.
    dtype : str, type, list of type or None, default='numeric'
        Data type of result.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : bool, default=False
        Whether a forced copy will be triggered.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan in array.
    ensure_2d : bool, default=True
        Whether to raise a value error if X is not 2D.
    allow_nd : bool, default=False
        Whether to allow X.ndim > 2.
    multi_output : bool, default=False
        Whether to allow 2D y.
    ensure_min_samples : int, default=1
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric dtype.
    estimator : str or estimator instance, default=None
        If passed, include name in error messages.

    Returns
    -------
    X_converted : ndarray or sparse matrix
        The converted and validated X.
    y_converted : ndarray
        The converted and validated y.
    """
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
    )
    
    if multi_output:
        y = check_array(
            y,
            accept_sparse=False,
            dtype=None if not y_numeric else dtype,
            order=order,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=False,
            allow_nd=False,
            estimator=estimator,
        )
    else:
        y = column_or_1d(y, warn=True)
        if y_numeric:
            y = check_array(
                y,
                accept_sparse=False,
                dtype=dtype,
                order=order,
                copy=copy,
                force_all_finite=force_all_finite,
                ensure_2d=False,
                allow_nd=False,
                estimator=estimator,
            )
    
    check_consistent_length(X, y)
    
    return X, y


def check_consistent_length(*arrays: Any) -> None:
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    *arrays : list or tuple of arrays
        Arrays to check.

    Raises
    ------
    ValueError
        If arrays have inconsistent lengths.
    """
    lengths = [len(X) if hasattr(X, "__len__") else X.shape[0] for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {lengths}")


def column_or_1d(y: Any, warn: bool = False) -> np.ndarray:
    """Reshape array to 1D if it's a column vector.

    Parameters
    ----------
    y : array-like
        Input array.
    warn : bool, default=False
        Whether to issue a warning if conversion is performed.

    Returns
    -------
    y_converted : ndarray
        The reshaped array.
    """
    y = np.asarray(y)
    shape = y.shape
    
    if len(shape) == 1:
        return y
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        return y.ravel()
    
    raise ValueError(f"y should be a 1d array, got an array of shape {shape} instead.")


def check_random_state(seed: Union[None, int, RandomState, np.random.Generator]) -> RandomState:
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, RandomState or Generator instance
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    random_state : RandomState
        The random state object.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return RandomState(seed)
    if isinstance(seed, RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        # Convert new-style generator to old-style RandomState
        return RandomState(seed.bit_generator._seed_seq.entropy)
    
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.RandomState instance"
    )


def _assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Check for NaN and infinity in array.

    Parameters
    ----------
    X : ndarray
        Input array.
    allow_nan : bool, default=False
        If True, do not throw error when NaN is present.

    Raises
    ------
    ValueError
        If array contains NaN or infinity.
    """
    if allow_nan:
        if np.isinf(X).any():
            raise ValueError("Input contains infinity.")
    else:
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinity.")


def has_fit_parameter(estimator: Any, parameter: str) -> bool:
    """Check whether the estimator's fit method supports a given parameter.

    Parameters
    ----------
    estimator : object
        An estimator to inspect.
    parameter : str
        The parameter to check.

    Returns
    -------
    is_parameter : bool
        Whether the parameter is supported.
    """
    import inspect
    
    if not hasattr(estimator, "fit"):
        return False
    
    fit_signature = inspect.signature(estimator.fit)
    return parameter in fit_signature.parameters


def check_classification_targets(y: np.ndarray) -> None:
    """Ensure that target y is of a classification type.

    Parameters
    ----------
    y : array-like
        Target values.

    Raises
    ------
    ValueError
        If y is not suitable for classification.
    """
    y_type = type_of_target(y)
    if y_type not in ["binary", "multiclass", "multiclass-multioutput", "multilabel-indicator"]:
        raise ValueError(f"Unknown label type: {y_type}")


def check_regression_targets(y: np.ndarray) -> None:
    """Ensure that target y is suitable for regression.

    Parameters
    ----------
    y : array-like
        Target values.

    Raises
    ------
    ValueError
        If y is not suitable for regression.
    """
    y_type = type_of_target(y)
    if y_type not in ["continuous", "continuous-multioutput"]:
        raise ValueError(f"Unknown label type: {y_type}")


def type_of_target(y: Any) -> str:
    """Determine the type of data indicated by the target.

    Parameters
    ----------
    y : array-like
        Target values.

    Returns
    -------
    target_type : str
        One of:
        * 'continuous': y is an array-like of floats that are not all integers.
        * 'continuous-multioutput': y is a 2d array of floats.
        * 'binary': y contains <= 2 discrete values.
        * 'multiclass': y contains more than 2 discrete values.
        * 'multiclass-multioutput': y is a 2d array with discrete values.
        * 'multilabel-indicator': y is a label indicator matrix.
        * 'unknown': can't determine the type.
    """
    y = np.asarray(y)
    
    # Check for sparse
    if sp.issparse(y):
        if y.format != "csr":
            y = y.tocsr()
        if len(y.shape) != 2:
            return "unknown"
        if y.shape[1] == 1:
            return "binary" if len(np.unique(y.data)) <= 2 else "multiclass"
        return "multilabel-indicator"
    
    # Dense array
    if y.ndim == 1:
        # Check if continuous or discrete
        if np.issubdtype(y.dtype, np.floating):
            return "continuous"
        unique_values = np.unique(y)
        if len(unique_values) <= 2:
            return "binary"
        return "multiclass"
    
    if y.ndim == 2:
        if y.shape[1] == 1:
            return type_of_target(y.ravel())
        # Check for multilabel
        if np.all((y == 0) | (y == 1)):
            return "multilabel-indicator"
        # Multioutput
        if np.issubdtype(y.dtype, np.floating):
            return "continuous-multioutput"
        return "multiclass-multioutput"
    
    return "unknown"


def check_symmetric(array: np.ndarray, tol: float = 1e-10, raise_exception: bool = True) -> bool:
    """Check if array is symmetric.

    Parameters
    ----------
    array : ndarray
        Input array.
    tol : float, default=1e-10
        Absolute tolerance for symmetry check.
    raise_exception : bool, default=True
        If True, raise an exception if not symmetric.

    Returns
    -------
    is_symmetric : bool
        True if array is symmetric within tolerance.

    Raises
    ------
    ValueError
        If array is not symmetric and raise_exception is True.
    """
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        if raise_exception:
            raise ValueError("Array must be square.")
        return False
    
    if sp.issparse(array):
        diff = array - array.T
        is_sym = np.abs(diff.data).max() < tol
    else:
        is_sym = np.allclose(array, array.T, atol=tol)
    
    if not is_sym and raise_exception:
        raise ValueError("Array is not symmetric.")
    
    return is_sym
