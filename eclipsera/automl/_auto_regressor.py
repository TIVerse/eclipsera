"""Automated regressor selection and tuning."""
from typing import Dict, List, Optional, Any

import numpy as np

from ..core.base import BaseRegressor
from ..core.validation import check_X_y, check_array
from ..ml import (
    LinearRegression,
    Ridge,
    Lasso,
    RandomForestRegressor,
    GradientBoostingRegressor,
    KNeighborsRegressor,
    DecisionTreeRegressor,
)
from ..model_selection import cross_val_score


class AutoRegressor(BaseRegressor):
    """Automatic regressor selection and optimization.
    
    Automatically tries multiple regression algorithms and selects
    the best performing one based on cross-validation.
    
    Parameters
    ----------
    scoring : str, default='r2'
        Metric to use for evaluation ('r2', 'neg_mean_squared_error').
    cv : int, default=5
        Number of cross-validation folds.
    algorithms : list of str, default=None
        List of algorithm names to try. If None, tries all available.
    verbose : int, default=1
        Verbosity level.
    random_state : int, default=None
        Random state for reproducibility.
        
    Attributes
    ----------
    best_estimator_ : estimator
        The best performing estimator.
    best_score_ : float
        Score of the best estimator.
    best_algorithm_ : str
        Name of the best algorithm.
    scores_ : dict
        Scores for all tried algorithms.
        
    Examples
    --------
    >>> from eclipsera.automl import AutoRegressor
    >>> auto_reg = AutoRegressor(cv=5, verbose=1)
    >>> auto_reg.fit(X_train, y_train)
    >>> y_pred = auto_reg.predict(X_test)
    """
    
    def __init__(
        self,
        scoring: str = 'r2',
        cv: int = 5,
        algorithms: Optional[List[str]] = None,
        verbose: int = 1,
        random_state: Optional[int] = None,
    ):
        self.scoring = scoring
        self.cv = cv
        self.algorithms = algorithms
        self.verbose = verbose
        self.random_state = random_state
    
    def _get_candidate_algorithms(self) -> Dict[str, Any]:
        """Get dictionary of candidate algorithms to try."""
        candidates = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
        }
        
        # Filter if specific algorithms requested
        if self.algorithms is not None:
            candidates = {k: v for k, v in candidates.items() if k in self.algorithms}
        
        return candidates
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoRegressor":
        """Find the best regressor for the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : AutoRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.n_features_in_ = X.shape[1]
        
        # Get candidate algorithms
        candidates = self._get_candidate_algorithms()
        
        if self.verbose:
            print(f"AutoRegressor: Evaluating {len(candidates)} algorithms...")
        
        # Evaluate each algorithm
        scores = {}
        
        for name, estimator in candidates.items():
            if self.verbose:
                print(f"  Trying {name}...", end=' ')
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    estimator, X, y,
                    cv=self.cv,
                    scoring=self.scoring
                )
                mean_score = np.mean(cv_scores)
                scores[name] = mean_score
                
                if self.verbose:
                    print(f"Score: {mean_score:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Failed: {str(e)}")
                scores[name] = -np.inf
        
        # Select best algorithm
        self.scores_ = scores
        self.best_algorithm_ = max(scores, key=scores.get)
        self.best_score_ = scores[self.best_algorithm_]
        
        if self.verbose:
            print(f"\nBest algorithm: {self.best_algorithm_} (score: {self.best_score_:.4f})")
        
        # Fit best estimator on full data
        self.best_estimator_ = candidates[self.best_algorithm_]
        self.best_estimator_.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("AutoRegressor must be fitted before calling predict")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using the best estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
            
        Returns
        -------
        score : float
            Score of the best estimator.
        """
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("AutoRegressor must be fitted before calling score")
        
        return self.best_estimator_.score(X, y)
