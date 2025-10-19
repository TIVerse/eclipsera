"""Feature importance extraction utilities."""
from typing import Optional, Dict

import numpy as np


def get_feature_importance(
    estimator,
    feature_names: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    """Extract feature importance from a fitted estimator.
    
    Works with estimators that have either `coef_` or `feature_importances_`
    attributes.
    
    Parameters
    ----------
    estimator : estimator object
        A fitted estimator with feature importance information.
    feature_names : list of str, default=None
        Names of features. If None, uses indices.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'importances': Feature importance values
        - 'feature_names': Feature names or indices
        - 'sorted_idx': Indices sorted by importance (descending)
        
    Examples
    --------
    >>> from eclipsera.explainability import get_feature_importance
    >>> from eclipsera.ml import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> clf.fit(X, y)
    >>> result = get_feature_importance(clf, feature_names=['age', 'income'])
    >>> print(result['importances'])
    """
    # Try to get feature importances
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    
    elif hasattr(estimator, 'coef_'):
        # For linear models, use absolute coefficients
        coef = estimator.coef_
        if coef.ndim > 1:
            # Multi-class: average absolute coefficients
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    
    else:
        raise AttributeError(
            "Estimator must have either 'feature_importances_' or 'coef_' attribute. "
            "Use permutation_importance() as an alternative."
        )
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    return {
        'importances': importances,
        'feature_names': feature_names,
        'sorted_idx': sorted_idx,
    }
