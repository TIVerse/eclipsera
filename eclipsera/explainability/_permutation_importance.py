"""Permutation importance for feature importance analysis."""
from typing import Optional, Dict

import numpy as np

from ..core.validation import check_array


def permutation_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Permutation importance for feature evaluation.
    
    Permutation importance measures the importance of a feature by calculating
    the increase in the prediction error after permuting the feature's values.
    
    Parameters
    ----------
    estimator : estimator object
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Data on which permutation importance will be computed.
    y : array-like of shape (n_samples,)
        Target values.
    scoring : str, default=None
        Scoring method. If None, uses estimator's score method.
    n_repeats : int, default=5
        Number of times to permute each feature.
    random_state : int, default=None
        Random state for reproducibility.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'importances_mean': Mean importance for each feature
        - 'importances_std': Standard deviation of importance
        - 'importances': All importance values (n_features, n_repeats)
        
    Examples
    --------
    >>> from eclipsera.explainability import permutation_importance
    >>> from eclipsera.ml import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> clf.fit(X_train, y_train)
    >>> result = permutation_importance(clf, X_test, y_test, n_repeats=10)
    >>> print(result['importances_mean'])
    """
    X = check_array(X)
    
    if not hasattr(estimator, 'predict'):
        raise ValueError("Estimator must have a predict method")
    
    n_samples, n_features = X.shape
    
    # Get baseline score
    baseline_score = estimator.score(X, y)
    
    # Initialize importance array
    importances = np.zeros((n_features, n_repeats))
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Compute importance for each feature
    for feature_idx in range(n_features):
        for repeat_idx in range(n_repeats):
            # Permute feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = rng.permutation(X_permuted[:, feature_idx])
            
            # Compute score with permuted feature
            permuted_score = estimator.score(X_permuted, y)
            
            # Importance is the decrease in score
            importances[feature_idx, repeat_idx] = baseline_score - permuted_score
    
    return {
        'importances': importances,
        'importances_mean': np.mean(importances, axis=1),
        'importances_std': np.std(importances, axis=1),
    }
