"""Train/test splitting and cross-validation utilities."""
from typing import Generator, Optional, Tuple, Union

import numpy as np

from ..core.utils import check_random_state
from ..core.validation import check_array


def train_test_split(
    *arrays,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None,
) -> list:
    """Split arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    *arrays : sequence of indexables
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices.
    test_size : float, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
    train_size : float, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        
    Returns
    -------
    splitting : list
        List containing train-test split of inputs.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.model_selection import train_test_split
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    >>> X_train.shape
    (2, 2)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    
    n_samples = len(arrays[0])
    
    # Validate all arrays have same length
    for array in arrays[1:]:
        if len(array) != n_samples:
            raise ValueError("All arrays must have the same length")
    
    # Determine sizes
    if test_size is None and train_size is None:
        test_size = 0.25
    
    if test_size is not None:
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be between 0 and 1")
        n_test = int(n_samples * test_size)
    else:
        if train_size < 0 or train_size > 1:
            raise ValueError("train_size must be between 0 and 1")
        n_train = int(n_samples * train_size)
        n_test = n_samples - n_train
    
    n_train = n_samples - n_test
    
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Train and test sets must be non-empty")
    
    # Generate indices
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Stratified split
        stratify = np.asarray(stratify)
        classes, y_indices = np.unique(stratify, return_inverse=True)
        
        train_indices = []
        test_indices = []
        
        random_state = check_random_state(random_state)
        
        for class_idx in range(len(classes)):
            class_mask = y_indices == class_idx
            class_indices = indices[class_mask]
            
            if shuffle:
                random_state.shuffle(class_indices)
            
            n_class_samples = len(class_indices)
            n_class_test = int(n_class_samples * (n_test / n_samples))
            
            test_indices.extend(class_indices[:n_class_test])
            train_indices.extend(class_indices[n_class_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
    else:
        # Random split
        if shuffle:
            random_state = check_random_state(random_state)
            random_state.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    # Split arrays
    result = []
    for array in arrays:
        array = np.asarray(array)
        result.append(array[train_indices])
        result.append(array[test_indices])
    
    return result


class KFold:
    """K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the ordering of the indices.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> kf = KFold(n_splits=2)
    >>> for train_idx, test_idx in kf.split(X):
    ...     print(train_idx, test_idx)
    [2 3] [0 1]
    [0 1] [2 3]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            The target variable (ignored).
        groups : array-like of shape (n_samples,), default=None
            Group labels (ignored).
            
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            random_state = check_random_state(self.random_state)
            random_state.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class StratifiedKFold:
    """Stratified K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage
    of samples for each class.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the ordering of the indices.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> for train_idx, test_idx in skf.split(X, y):
    ...     print(train_idx, test_idx)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels (ignored).
            
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        y = np.asarray(y)
        n_samples = len(y)
        
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        
        # Sort by class
        class_counts = np.bincount(y_indices)
        
        # Check that we have enough samples per class
        if np.any(class_counts < self.n_splits):
            raise ValueError(
                f"The least populated class has {class_counts.min()} members, "
                f"which is less than n_splits={self.n_splits}."
            )
        
        # Shuffle within each class
        random_state = check_random_state(self.random_state) if self.shuffle else None
        
        # Build test sets for each fold
        test_folds = [[] for _ in range(self.n_splits)]
        
        for class_idx in range(n_classes):
            class_mask = y_indices == class_idx
            class_indices = np.where(class_mask)[0]
            
            if self.shuffle and random_state is not None:
                random_state.shuffle(class_indices)
            
            # Distribute class samples across folds
            fold_sizes = np.full(self.n_splits, len(class_indices) // self.n_splits, dtype=int)
            fold_sizes[:len(class_indices) % self.n_splits] += 1
            
            current = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                test_folds[fold_idx].extend(class_indices[current:current + fold_size])
                current += fold_size
        
        # Generate train/test splits
        all_indices = np.arange(n_samples)
        for test_indices in test_folds:
            test_indices = np.array(test_indices)
            train_indices = np.setdiff1d(all_indices, test_indices)
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


def cross_val_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Optional[Union[int, object]] = None,
    scoring: Optional[str] = None,
) -> np.ndarray:
    """Evaluate a score by cross-validation.
    
    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        If int, specifies the number of folds in a KFold.
    scoring : str, default=None
        Scoring method to use. If None, uses the estimator's score method.
        
    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        
    Examples
    --------
    >>> from eclipsera.ml import LogisticRegression
    >>> from eclipsera.model_selection import cross_val_score
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [0, 0, 1, 1]
    >>> clf = LogisticRegression()
    >>> scores = cross_val_score(clf, X, y, cv=2)
    >>> scores.mean()
    """
    X = check_array(X)
    y = check_array(y, ensure_2d=False)
    
    # Default cv
    if cv is None:
        cv = 5
    
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    
    scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and score
        estimator_clone = estimator.__class__(**estimator.get_params())
        estimator_clone.fit(X_train, y_train)
        
        if scoring is None:
            score = estimator_clone.score(X_test, y_test)
        else:
            # For now, just use score method
            # TODO: Add support for different scoring functions
            score = estimator_clone.score(X_test, y_test)
        
        scores.append(score)
    
    return np.array(scores)


def cross_validate(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Optional[Union[int, object]] = None,
    scoring: Optional[Union[str, list]] = None,
    return_train_score: bool = False,
) -> dict:
    """Evaluate metric(s) by cross-validation and return various scores.
    
    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str or list, default=None
        Scoring method(s) to use.
    return_train_score : bool, default=False
        Whether to include train scores.
        
    Returns
    -------
    scores : dict
        A dict with timing and scoring information.
        
    Examples
    --------
    >>> from eclipsera.ml import LogisticRegression
    >>> from eclipsera.model_selection import cross_validate
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [0, 0, 1, 1]
    >>> clf = LogisticRegression()
    >>> cv_results = cross_validate(clf, X, y, cv=2)
    >>> cv_results['test_score']
    """
    X = check_array(X)
    y = check_array(y, ensure_2d=False)
    
    if cv is None:
        cv = 5
    
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    
    test_scores = []
    train_scores = [] if return_train_score else None
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit
        estimator_clone = estimator.__class__(**estimator.get_params())
        estimator_clone.fit(X_train, y_train)
        
        # Test score
        test_score = estimator_clone.score(X_test, y_test)
        test_scores.append(test_score)
        
        # Train score if requested
        if return_train_score:
            train_score = estimator_clone.score(X_train, y_train)
            train_scores.append(train_score)
    
    result = {
        'test_score': np.array(test_scores),
    }
    
    if return_train_score:
        result['train_score'] = np.array(train_scores)
    
    return result
