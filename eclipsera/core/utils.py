"""Core utility functions for Eclipsera."""
import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.random import RandomState

from .exceptions import DataDimensionalityError
from .validation import check_array, check_random_state as _check_random_state


def safe_sparse_dot(
    a: Union[np.ndarray, sp.spmatrix],
    b: Union[np.ndarray, sp.spmatrix],
    dense_output: bool = False,
) -> Union[np.ndarray, sp.spmatrix]:
    """Dot product that handles sparse matrices correctly.

    Parameters
    ----------
    a : array or sparse matrix
        First array.
    b : array or sparse matrix
        Second array.
    dense_output : bool, default=False
        If True, return dense output even if inputs are sparse.

    Returns
    -------
    dot_product : array or sparse matrix
        The dot product of a and b.
    """
    if sp.issparse(a) or sp.issparse(b):
        ret = a @ b
        if dense_output and sp.issparse(ret):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def check_random_state(seed: Union[None, int, RandomState]) -> RandomState:
    """Turn seed into a RandomState instance.

    Parameters
    ----------
    seed : None, int or RandomState
        Seed for the random number generator.

    Returns
    -------
    random_state : RandomState
        The random state object.
    """
    return _check_random_state(seed)


def safe_indexing(X: Union[np.ndarray, list], indices: np.ndarray) -> Union[np.ndarray, list]:
    """Return items or rows from X using indices.

    Parameters
    ----------
    X : array-like, sparse matrix, or list
        Data from which to sample.
    indices : array-like of int
        Indices to select.

    Returns
    -------
    subset : array-like, sparse matrix, or list
        Selected subset of X.
    """
    if hasattr(X, "iloc"):
        # pandas DataFrame or Series
        return X.iloc[indices]
    elif sp.issparse(X):
        return X[indices]
    elif isinstance(X, list):
        return [X[i] for i in indices]
    else:
        return X[indices]


def shuffle(*arrays: Any, random_state: Optional[Union[int, RandomState]] = None, n_samples: Optional[int] = None) -> Union[np.ndarray, tuple]:
    """Shuffle arrays in a consistent way.

    Parameters
    ----------
    *arrays : sequence of arrays
        Arrays to shuffle. Must have same first dimension.
    random_state : int, RandomState or None, default=None
        Random state for shuffling.
    n_samples : int or None, default=None
        Number of samples to generate. If None, use all samples.

    Returns
    -------
    shuffled_arrays : array or list of arrays
        Shuffled arrays.
    """
    if len(arrays) == 0:
        return None
    
    random_state = check_random_state(random_state)
    n = len(arrays[0])
    
    if n_samples is None:
        n_samples = n
    
    indices = random_state.permutation(n)[:n_samples]
    
    if len(arrays) == 1:
        return safe_indexing(arrays[0], indices)
    else:
        return tuple(safe_indexing(arr, indices) for arr in arrays)


def resample(*arrays: Any, replace: bool = True, n_samples: Optional[int] = None, random_state: Optional[Union[int, RandomState]] = None, stratify: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
    """Resample arrays in a consistent way.

    Parameters
    ----------
    *arrays : sequence of arrays
        Arrays to resample.
    replace : bool, default=True
        Whether to sample with replacement.
    n_samples : int or None, default=None
        Number of samples to generate.
    random_state : int, RandomState or None, default=None
        Random state for resampling.
    stratify : array-like or None, default=None
        If not None, data is split in a stratified fashion using this as class labels.

    Returns
    -------
    resampled_arrays : array or list of arrays
        Resampled arrays.
    """
    random_state = check_random_state(random_state)
    n = len(arrays[0])
    
    if n_samples is None:
        n_samples = n
    
    if stratify is not None:
        # Stratified resampling
        from collections import Counter
        class_counts = Counter(stratify)
        indices = []
        
        for class_label in class_counts:
            class_indices = np.where(stratify == class_label)[0]
            n_class_samples = int(n_samples * len(class_indices) / n)
            class_resampled = random_state.choice(
                class_indices, size=n_class_samples, replace=replace
            )
            indices.extend(class_resampled)
        
        indices = np.array(indices)
        random_state.shuffle(indices)
    else:
        if replace:
            indices = random_state.randint(0, n, size=n_samples)
        else:
            indices = random_state.permutation(n)[:n_samples]
    
    if len(arrays) == 1:
        return safe_indexing(arrays[0], indices)
    else:
        return tuple(safe_indexing(arr, indices) for arr in arrays)


def compute_sample_weight(class_weight: Union[dict, str, None], y: np.ndarray) -> np.ndarray:
    """Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        Weights associated with classes.
    y : array-like
        Class labels.

    Returns
    -------
    sample_weight : ndarray
        Array with sample weights.
    """
    if class_weight is None:
        return np.ones(len(y), dtype=np.float64)
    
    classes = np.unique(y)
    
    if class_weight == "balanced":
        # Compute balanced class weights
        n_samples = len(y)
        n_classes = len(classes)
        class_weight_dict = {}
        
        for c in classes:
            class_weight_dict[c] = n_samples / (n_classes * np.sum(y == c))
    elif isinstance(class_weight, dict):
        class_weight_dict = class_weight
    else:
        raise ValueError("class_weight must be dict, 'balanced', or None")
    
    # Map class weights to samples
    sample_weight = np.array([class_weight_dict.get(c, 1.0) for c in y])
    
    return sample_weight


def softmax(X: np.ndarray, copy: bool = True) -> np.ndarray:
    """Compute the softmax function in a numerically stable way.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input array.
    copy : bool, default=True
        Whether to perform the operation in-place or copy.

    Returns
    -------
    softmax_array : ndarray of shape (n_samples, n_features)
        Softmax probabilities.
    """
    # Always work with float arrays
    X = np.array(X, dtype=np.float64, copy=copy)
    
    # Subtract max for numerical stability
    max_vals = np.max(X, axis=1, keepdims=True)
    X -= max_vals
    
    np.exp(X, out=X)
    sum_exp = np.sum(X, axis=1, keepdims=True)
    X /= sum_exp
    
    return X


def log_softmax(X: np.ndarray) -> np.ndarray:
    """Compute log of softmax in a numerically stable way.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input array.

    Returns
    -------
    log_softmax_array : ndarray of shape (n_samples, n_features)
        Log softmax values.
    """
    max_vals = np.max(X, axis=1, keepdims=True)
    X_shifted = X - max_vals
    log_sum_exp = np.log(np.sum(np.exp(X_shifted), axis=1, keepdims=True))
    
    return X_shifted - log_sum_exp


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15, sample_weight: Optional[np.ndarray] = None) -> float:
    """Log loss, aka logistic loss or cross-entropy loss.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Predicted probabilities.
    eps : float, default=1e-15
        Small value to clip probabilities.
    sample_weight : array-like or None, default=None
        Sample weights.

    Returns
    -------
    loss : float
        Log loss value.
    """
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    if y_prob.ndim == 1:
        # Binary classification
        loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    else:
        # Multiclass
        loss = -np.sum(y_true * np.log(y_prob), axis=1)
    
    if sample_weight is not None:
        loss = loss * sample_weight
    
    return np.mean(loss)


def safe_sqr(X: np.ndarray, copy: bool = True) -> np.ndarray:
    """Element-wise squaring of array.

    Parameters
    ----------
    X : array-like
        Input array.
    copy : bool, default=True
        Whether to copy the array.

    Returns
    -------
    X_sqr : ndarray
        Squared array.
    """
    if copy:
        X = np.array(X, copy=True)
    return np.multiply(X, X, out=X)


def parallel_helper(func: callable, iterable: Any, n_jobs: int = -1, **kwargs: Any) -> list:
    """Helper function for parallel execution.

    Parameters
    ----------
    func : callable
        Function to execute in parallel.
    iterable : iterable
        Items to process.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all processors.
    **kwargs : dict
        Additional keyword arguments for Parallel.

    Returns
    -------
    results : list
        Results from parallel execution.
    """
    return Parallel(n_jobs=n_jobs, **kwargs)(delayed(func)(item) for item in iterable)


def get_chunk_n_rows(row_bytes: int, max_n_rows: Optional[int] = None, working_memory: float = 1024.0) -> int:
    """Calculate how many rows can be processed in a chunk.

    Parameters
    ----------
    row_bytes : int
        Number of bytes per row.
    max_n_rows : int or None, default=None
        Maximum number of rows.
    working_memory : float, default=1024.0
        Memory budget in MB.

    Returns
    -------
    chunk_n_rows : int
        Number of rows that can fit in memory.
    """
    chunk_n_rows = int(working_memory * 1024 * 1024 // row_bytes)
    
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    
    return max(1, chunk_n_rows)


def check_scalar(
    x: Any,
    name: str,
    target_type: Union[type, tuple],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    include_boundaries: str = "both",
) -> None:
    """Validate scalar parameters.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.
    name : str
        The name of the parameter.
    target_type : type or tuple of types
        Acceptable types for the parameter.
    min_val : float or None, default=None
        Minimum acceptable value.
    max_val : float or None, default=None
        Maximum acceptable value.
    include_boundaries : {'left', 'right', 'both', 'neither'}, default='both'
        Whether to include boundaries in the range.

    Raises
    ------
    TypeError
        If the parameter is not of an acceptable type.
    ValueError
        If the parameter is outside the acceptable range.
    """
    if not isinstance(x, target_type):
        raise TypeError(f"{name} must be an instance of {target_type}, got {type(x)}")
    
    if min_val is not None or max_val is not None:
        if min_val is not None:
            if include_boundaries in ("left", "both"):
                if x < min_val:
                    raise ValueError(f"{name} must be >= {min_val}, got {x}")
            else:
                if x <= min_val:
                    raise ValueError(f"{name} must be > {min_val}, got {x}")
        
        if max_val is not None:
            if include_boundaries in ("right", "both"):
                if x > max_val:
                    raise ValueError(f"{name} must be <= {max_val}, got {x}")
            else:
                if x >= max_val:
                    raise ValueError(f"{name} must be < {max_val}, got {x}")
