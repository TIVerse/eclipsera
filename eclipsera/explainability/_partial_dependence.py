"""Partial dependence for understanding feature effects."""
from typing import Optional, Dict, List, Tuple

import numpy as np

from ..core.validation import check_array


def partial_dependence(
    estimator,
    X: np.ndarray,
    features: List[int],
    grid_resolution: int = 100,
    percentiles: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, np.ndarray]:
    """Partial dependence of features.
    
    Partial dependence shows the dependence between the target and a set
    of features, marginalizing over the values of all other features.
    
    Parameters
    ----------
    estimator : estimator object
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Data to use for partial dependence.
    features : list of int
        Indices of features for which to compute partial dependence.
    grid_resolution : int, default=100
        Number of points on the grid for each feature.
    percentiles : tuple of float, default=(0.05, 0.95)
        Lower and upper percentiles for grid range.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'values': List of grid values for each feature
        - 'predictions': Partial dependence values
        - 'features': Feature indices
        
    Examples
    --------
    >>> from eclipsera.explainability import partial_dependence
    >>> result = partial_dependence(clf, X, features=[0, 1])
    >>> print(result['predictions'][0])  # PD for feature 0
    """
    X = check_array(X)
    
    if not hasattr(estimator, 'predict'):
        raise ValueError("Estimator must have a predict method")
    
    n_samples, n_features = X.shape
    
    # Prepare results
    grid_values = []
    pd_values = []
    
    for feature_idx in features:
        # Create grid for this feature
        feature_min = np.percentile(X[:, feature_idx], percentiles[0] * 100)
        feature_max = np.percentile(X[:, feature_idx], percentiles[1] * 100)
        grid = np.linspace(feature_min, feature_max, grid_resolution)
        
        # Compute partial dependence
        pd = np.zeros(grid_resolution)
        
        for i, grid_value in enumerate(grid):
            # Create dataset with feature set to grid value
            X_temp = X.copy()
            X_temp[:, feature_idx] = grid_value
            
            # Predict and average
            predictions = estimator.predict(X_temp)
            pd[i] = np.mean(predictions)
        
        grid_values.append(grid)
        pd_values.append(pd)
    
    return {
        'values': grid_values,
        'predictions': pd_values,
        'features': features,
    }


def plot_partial_dependence(
    estimator,
    X: np.ndarray,
    features: List[int],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 100,
):
    """Plot partial dependence.
    
    Parameters
    ----------
    estimator : estimator object
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Data to use for partial dependence.
    features : list of int
        Indices of features to plot.
    feature_names : list of str, default=None
        Names of features for plot labels.
    grid_resolution : int, default=100
        Number of points on the grid.
        
    Returns
    -------
    result : dict
        Partial dependence result dictionary.
        
    Examples
    --------
    >>> from eclipsera.explainability import plot_partial_dependence
    >>> result = plot_partial_dependence(clf, X, features=[0, 1, 2])
    """
    result = partial_dependence(estimator, X, features, grid_resolution)
    
    try:
        import matplotlib.pyplot as plt
        
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, (feature_idx, grid, pd) in enumerate(zip(
            features, result['values'], result['predictions']
        )):
            axes[idx].plot(grid, pd, linewidth=2)
            
            if feature_names is not None:
                axes[idx].set_xlabel(feature_names[feature_idx])
            else:
                axes[idx].set_xlabel(f'Feature {feature_idx}')
            
            axes[idx].set_ylabel('Partial Dependence')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    except ImportError:
        print("matplotlib not available for plotting")
    
    return result
