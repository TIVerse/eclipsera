"""Base classes for all estimators in Eclipsera.

This module provides the foundational base classes that all estimators inherit from,
defining the common API and behavior across the framework.
"""
import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import numpy as np

from .exceptions import NotFittedError, check_is_fitted


class BaseEstimator(metaclass=ABCMeta):
    """Base class for all estimators in Eclipsera.

    All estimators should inherit from this class and implement the fit method.
    The class provides common functionality like parameter getting/setting,
    cloning, and string representation.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Examples
    --------
    >>> from eclipsera.core.base import BaseEstimator
    >>> class MyEstimator(BaseEstimator):
    ...     def __init__(self, param1=1, param2=2):
    ...         self.param1 = param1
    ...         self.param2 = param2
    ...
    ...     def fit(self, X, y=None):
    ...         self.fitted_param_ = self.param1 + self.param2
    ...         return self
    >>> est = MyEstimator(param1=10)
    >>> est.get_params()
    {'param1': 10, 'param2': 2}
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        param_names : list of str
            List of parameter names.
        """
        # Get the constructor signature
        init_signature = inspect.signature(cls.__init__)
        
        # Extract parameters, excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        
        # Sort parameters alphabetically
        param_names = sorted([p.name for p in parameters])
        
        return param_names

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                # Recursively get params for nested estimators
                deep_items = value.get_params(deep=True).items()
                params.update((key + "__" + k, val) for k, val in deep_items)
            params[key] = value
        
        return params

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : BaseEstimator
            Estimator instance.

        Raises
        ------
        ValueError
            If invalid parameters are provided.
        """
        if not params:
            return self
        
        valid_params = self.get_params(deep=True)
        
        nested_params = {}
        for key, value in params.items():
            if "__" in key:
                # Handle nested parameters
                key_parts = key.split("__", 1)
                if key_parts[0] not in nested_params:
                    nested_params[key_parts[0]] = {}
                nested_params[key_parts[0]][key_parts[1]] = value
            else:
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                        f"Valid parameters are: {list(valid_params.keys())}"
                    )
                setattr(self, key, value)
        
        # Set nested parameters
        for key, sub_params in nested_params.items():
            if hasattr(getattr(self, key), "set_params"):
                getattr(self, key).set_params(**sub_params)
        
        return self

    def __repr__(self) -> str:
        """Return string representation of the estimator.

        Returns
        -------
        repr_str : str
            String representation.
        """
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        
        # Format parameters
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                param_strs.append(f"{key}={value!r}")
            else:
                param_strs.append(f"{key}={value}")
        
        return f"{class_name}({', '.join(param_strs)})"

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling.

        Returns
        -------
        state : dict
            Object state.
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state for unpickling.

        Parameters
        ----------
        state : dict
            Object state.
        """
        self.__dict__.update(state)


def clone(estimator: BaseEstimator, safe: bool = True) -> BaseEstimator:
    """Construct a new unfitted estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It returns a new estimator
    with the same parameters that has not been fitted on any data.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be cloned.
    safe : bool, default=True
        If True, raise an error if the estimator cannot be cloned.

    Returns
    -------
    estimator_clone : BaseEstimator
        The cloned estimator.

    Examples
    --------
    >>> from eclipsera.ml.linear import LinearRegression
    >>> from eclipsera.core.base import clone
    >>> model = LinearRegression(fit_intercept=True)
    >>> model_clone = clone(model)
    >>> model_clone.get_params()
    {'fit_intercept': True}
    """
    estimator_type = type(estimator)
    
    # Check if estimator has get_params
    if not hasattr(estimator, "get_params"):
        if safe:
            raise TypeError(
                f"Cannot clone object {estimator!r} (type {estimator_type}): "
                "it does not have a get_params method."
            )
        return copy.deepcopy(estimator)
    
    # Get parameters and create new instance
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    
    # Clone nested estimators
    for name, param in new_object_params.items():
        if hasattr(param, "get_params"):
            new_object_params[name] = clone(param, safe=safe)
    
    new_object = klass(**new_object_params)
    
    # Copy additional attributes if needed
    params_set = new_object.get_params(deep=False)
    
    return new_object


class ClassifierMixin:
    """Mixin class for all classifiers in Eclipsera."""

    _estimator_type = "classifier"

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics import accuracy_score
        
        predictions = self.predict(X)
        return accuracy_score(y, predictions, sample_weight=sample_weight)


class RegressorMixin:
    """Mixin class for all regressors in Eclipsera."""

    _estimator_type = "regressor"

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from .metrics import r2_score
        
        predictions = self.predict(X)
        return r2_score(y, predictions, sample_weight=sample_weight)


class TransformerMixin:
    """Mixin class for all transformers in Eclipsera."""

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y, **fit_params).transform(X)


class ClusterMixin:
    """Mixin class for all clustering estimators in Eclipsera."""

    _estimator_type = "clusterer"

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform clustering on X and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_


class MetaEstimatorMixin:
    """Mixin class for all meta estimators in Eclipsera.

    A meta estimator is an estimator that wraps one or more other estimators.
    """

    pass


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """Abstract base class for classifiers.

    All classifier implementations should inherit from this class and
    implement fit, predict, and optionally predict_proba.

    Examples
    --------
    >>> from eclipsera.core.base import BaseClassifier
    >>> import numpy as np
    >>> class DummyClassifier(BaseClassifier):
    ...     def __init__(self, strategy='constant', constant=0):
    ...         self.strategy = strategy
    ...         self.constant = constant
    ...
    ...     def fit(self, X, y):
    ...         self.classes_ = np.unique(y)
    ...         self.n_classes_ = len(self.classes_)
    ...         return self
    ...
    ...     def predict(self, X):
    ...         n_samples = X.shape[0]
    ...         return np.full(n_samples, self.constant)
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseClassifier":
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        self : BaseClassifier
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        pass

    def explain(self, X: np.ndarray, method: str = "auto", **kwargs: Any) -> Any:
        """Explain predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.
        method : str, default='auto'
            Explanation method to use.
        **kwargs : dict
            Additional parameters for the explanation method.

        Returns
        -------
        explanations : object
            Model explanations.
        """
        # Placeholder for explanation functionality
        # Will be implemented in explain module
        raise NotImplementedError(
            f"Explanation not yet implemented for {self.__class__.__name__}. "
            "Use eclipsera.explain module for model-agnostic explanations."
        )


class BaseRegressor(BaseEstimator, RegressorMixin):
    """Abstract base class for regressors.

    All regressor implementations should inherit from this class and
    implement fit and predict.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseRegressor":
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        self : BaseRegressor
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        pass

    def explain(self, X: np.ndarray, method: str = "auto", **kwargs: Any) -> Any:
        """Explain predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.
        method : str, default='auto'
            Explanation method to use.
        **kwargs : dict
            Additional parameters for the explanation method.

        Returns
        -------
        explanations : object
            Model explanations.
        """
        raise NotImplementedError(
            f"Explanation not yet implemented for {self.__class__.__name__}. "
            "Use eclipsera.explain module for model-agnostic explanations."
        )


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Abstract base class for transformers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> "BaseTransformer":
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        self : BaseTransformer
            Fitted transformer.
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_new)
            Transformed data.
        """
        pass


class BaseCluster(BaseEstimator, ClusterMixin):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> "BaseCluster":
        """Fit the clustering algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        self : BaseCluster
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        pass
