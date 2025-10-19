"""Hyperparameter search with cross-validation."""
from itertools import product
from typing import Any, Dict, Optional, Union

import numpy as np

from ..core.utils import check_random_state
from ..core.validation import check_array
from ._split import KFold, StratifiedKFold


class GridSearchCV:
    """Exhaustive search over specified parameter values for an estimator.
    
    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    param_grid : dict or list of dicts
        Dictionary with parameters names as keys and lists of parameter
        settings to try as values.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str, default=None
        Strategy to evaluate the performance of the cross-validated model.
    n_jobs : int, default=None
        Number of jobs to run in parallel (not implemented yet).
    refit : bool, default=True
        Refit an estimator using the best found parameters.
    verbose : int, default=0
        Controls the verbosity.
        
    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search.
    best_score_ : float
        Mean cross-validated score of the best_estimator.
    best_params_ : dict
        Parameter setting that gave the best results.
    cv_results_ : dict
        Dictionary with keys as column headers and values as columns.
        
    Examples
    --------
    >>> from eclipsera.ml import LogisticRegression
    >>> from eclipsera.model_selection import GridSearchCV
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]] * 5
    >>> y = [0, 0, 1, 1] * 5
    >>> param_grid = {'C': [0.1, 1.0, 10.0]}
    >>> clf = GridSearchCV(LogisticRegression(), param_grid, cv=3)
    >>> clf.fit(X, y)
    GridSearchCV(...)
    >>> clf.best_params_
    {'C': 1.0}
    """
    
    def __init__(
        self,
        estimator,
        param_grid: Union[Dict[str, list], list],
        cv: Optional[Union[int, object]] = None,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
    
    def _get_param_iterator(self):
        """Get parameter iterator from param_grid."""
        if isinstance(self.param_grid, dict):
            items = sorted(self.param_grid.items())
            keys, values = zip(*items)
            for v in product(*values):
                yield dict(zip(keys, v))
        elif isinstance(self.param_grid, list):
            for grid in self.param_grid:
                items = sorted(grid.items())
                keys, values = zip(*items)
                for v in product(*values):
                    yield dict(zip(keys, v))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Run fit with all sets of parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : GridSearchCV
            Fitted estimator.
        """
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        
        # Set up cross-validation
        if self.cv is None:
            cv = 5
        elif isinstance(self.cv, int):
            cv = self.cv
        else:
            cv = self.cv
        
        # Determine if we should use stratified CV
        if hasattr(self.estimator, 'classes_') or hasattr(self.estimator, '_estimator_type'):
            if isinstance(self.cv, int) or self.cv is None:
                cv_splitter = StratifiedKFold(n_splits=cv if isinstance(cv, int) else 5)
            else:
                cv_splitter = self.cv
        else:
            if isinstance(self.cv, int) or self.cv is None:
                cv_splitter = KFold(n_splits=cv if isinstance(cv, int) else 5)
            else:
                cv_splitter = self.cv
        
        # Grid search
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'test_scores': [],
        }
        
        best_score = -np.inf
        best_params = None
        best_estimator = None
        
        for params in self._get_param_iterator():
            if self.verbose > 0:
                print(f"Fitting with params: {params}")
            
            # Cross-validate with these parameters
            scores = []
            
            for train_idx, test_idx in cv_splitter.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Create and fit estimator
                estimator = self.estimator.__class__(**{**self.estimator.get_params(), **params})
                estimator.fit(X_train, y_train)
                
                # Score
                if self.scoring is None:
                    score = estimator.score(X_test, y_test)
                else:
                    # For now, just use score method
                    score = estimator.score(X_test, y_test)
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            results['test_scores'].append(scores)
            
            if self.verbose > 0:
                print(f"  Mean score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Track best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = estimator
        
        self.cv_results_ = results
        self.best_score_ = best_score
        self.best_params_ = best_params
        
        # Refit on full data if requested
        if self.refit:
            self.best_estimator_ = self.estimator.__class__(
                **{**self.estimator.get_params(), **best_params}
            )
            self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_ = best_estimator
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Call predict on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("GridSearchCV must be fitted before calling predict")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the score on the given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns
        -------
        score : float
            Score of best_estimator on the data.
        """
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("GridSearchCV must be fitted before calling score")
        
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV:
    """Randomized search on hyper parameters.
    
    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba" if they are implemented
    in the estimator used.
    
    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    param_distributions : dict
        Dictionary with parameters names as keys and distributions or lists
        of parameters to try.
    n_iter : int, default=10
        Number of parameter settings that are sampled.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str, default=None
        Strategy to evaluate the performance of the cross-validated model.
    n_jobs : int, default=None
        Number of jobs to run in parallel (not implemented yet).
    refit : bool, default=True
        Refit an estimator using the best found parameters.
    verbose : int, default=0
        Controls the verbosity.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state.
        
    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search.
    best_score_ : float
        Mean cross-validated score of the best_estimator.
    best_params_ : dict
        Parameter setting that gave the best results.
    cv_results_ : dict
        Dictionary with keys as column headers and values as columns.
        
    Examples
    --------
    >>> from eclipsera.ml import LogisticRegression
    >>> from eclipsera.model_selection import RandomizedSearchCV
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]] * 5
    >>> y = [0, 0, 1, 1] * 5
    >>> param_dist = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}
    >>> clf = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=3, cv=3)
    >>> clf.fit(X, y)
    RandomizedSearchCV(...)
    """
    
    def __init__(
        self,
        estimator,
        param_distributions: Dict[str, list],
        n_iter: int = 10,
        cv: Optional[Union[int, object]] = None,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
    
    def _sample_parameters(self, random_state):
        """Sample n_iter candidates from param_distributions."""
        for _ in range(self.n_iter):
            params = {}
            for key, value in self.param_distributions.items():
                if isinstance(value, list):
                    # Sample from list
                    params[key] = random_state.choice(value)
                else:
                    # Assume it's a distribution-like object with rvs method
                    if hasattr(value, 'rvs'):
                        params[key] = value.rvs(random_state=random_state)
                    else:
                        params[key] = value
            yield params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomizedSearchCV":
        """Run fit with all sets of parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : RandomizedSearchCV
            Fitted estimator.
        """
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        
        random_state = check_random_state(self.random_state)
        
        # Set up cross-validation
        if self.cv is None:
            cv = 5
        elif isinstance(self.cv, int):
            cv = self.cv
        else:
            cv = self.cv
        
        # Determine if we should use stratified CV
        if hasattr(self.estimator, 'classes_') or hasattr(self.estimator, '_estimator_type'):
            if isinstance(self.cv, int) or self.cv is None:
                cv_splitter = StratifiedKFold(n_splits=cv if isinstance(cv, int) else 5)
            else:
                cv_splitter = self.cv
        else:
            if isinstance(self.cv, int) or self.cv is None:
                cv_splitter = KFold(n_splits=cv if isinstance(cv, int) else 5)
            else:
                cv_splitter = self.cv
        
        # Randomized search
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'test_scores': [],
        }
        
        best_score = -np.inf
        best_params = None
        best_estimator = None
        
        for params in self._sample_parameters(random_state):
            if self.verbose > 0:
                print(f"Fitting with params: {params}")
            
            # Cross-validate with these parameters
            scores = []
            
            for train_idx, test_idx in cv_splitter.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Create and fit estimator
                estimator = self.estimator.__class__(**{**self.estimator.get_params(), **params})
                estimator.fit(X_train, y_train)
                
                # Score
                if self.scoring is None:
                    score = estimator.score(X_test, y_test)
                else:
                    score = estimator.score(X_test, y_test)
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            results['test_scores'].append(scores)
            
            if self.verbose > 0:
                print(f"  Mean score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Track best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = estimator
        
        self.cv_results_ = results
        self.best_score_ = best_score
        self.best_params_ = best_params
        
        # Refit on full data if requested
        if self.refit:
            self.best_estimator_ = self.estimator.__class__(
                **{**self.estimator.get_params(), **best_params}
            )
            self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_ = best_estimator
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Call predict on the estimator with the best found parameters."""
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("RandomizedSearchCV must be fitted before calling predict")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the score on the given data."""
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("RandomizedSearchCV must be fitted before calling score")
        
        return self.best_estimator_.score(X, y)
