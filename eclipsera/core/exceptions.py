"""Exception hierarchy for Eclipsera.

This module defines custom exceptions and warnings used throughout the framework.
"""
from typing import Any, Optional


class EclipseraException(Exception):
    """Base exception class for all Eclipsera exceptions.

    All custom exceptions in Eclipsera inherit from this class to allow
    easy catching of framework-specific errors.
    """

    pass


class NotFittedError(EclipseraException, ValueError, AttributeError):
    """Exception raised when an estimator is used before being fitted.

    This exception is raised when a method that requires a fitted estimator
    is called before the estimator's fit() method has been invoked.

    Examples
    --------
    >>> from eclipsera.ml.linear import LinearRegression
    >>> model = LinearRegression()
    >>> try:
    ...     model.predict([[1, 2, 3]])
    ... except NotFittedError as e:
    ...     print("Model not fitted!")
    Model not fitted!
    """

    pass


class InvalidParameterError(EclipseraException, ValueError):
    """Exception raised when an invalid parameter value is provided.

    This exception is raised during parameter validation when a parameter
    value is outside the acceptable range or of an incorrect type.

    Parameters
    ----------
    message : str
        Detailed error message describing the invalid parameter.
    """

    pass


class DataDimensionalityError(EclipseraException, ValueError):
    """Exception raised when input data has incorrect dimensions.

    This exception is raised when the dimensionality of input arrays
    doesn't match the expected shape or is incompatible with the model.

    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.linear import LinearRegression
    >>> model = LinearRegression()
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> model.fit(X, y)
    >>> X_new = np.random.randn(10, 3)  # Wrong number of features
    >>> try:
    ...     model.predict(X_new)
    ... except DataDimensionalityError:
    ...     print("Dimension mismatch!")
    Dimension mismatch!
    """

    pass


class FeatureNamesError(EclipseraException, ValueError):
    """Exception raised when feature names are inconsistent.

    This exception is raised when feature names between training and
    prediction differ, or when feature name constraints are violated.
    """

    pass


class NotSupportedError(EclipseraException, NotImplementedError):
    """Exception raised when a feature is not supported.

    This exception is raised when attempting to use a feature that is
    not implemented for the current configuration or estimator type.
    """

    pass


class ModelNotLoadedError(EclipseraException, RuntimeError):
    """Exception raised when attempting to use a model that hasn't been loaded.

    This exception is raised when operations require a loaded model
    but the model hasn't been loaded from disk or initialized properly.
    """

    pass


class ValidationError(EclipseraException, ValueError):
    """Exception raised when data validation fails.

    This exception is raised during input validation when data doesn't
    meet the required criteria (e.g., contains NaN, infinite values,
    or violates other constraints).
    """

    pass


class ConvergenceWarning(UserWarning):
    """Warning issued when an algorithm fails to converge.

    This warning is issued when an iterative algorithm doesn't converge
    within the specified number of iterations or tolerance threshold.

    Examples
    --------
    >>> import warnings
    >>> from eclipsera.core.exceptions import ConvergenceWarning
    >>> warnings.warn("Algorithm did not converge", ConvergenceWarning)
    """

    pass


class FitFailedWarning(UserWarning):
    """Warning issued when a fit operation fails.

    This warning is issued when a model fitting procedure encounters
    an error but execution can continue (e.g., in cross-validation
    where some folds may fail).
    """

    pass


class DataConversionWarning(UserWarning):
    """Warning issued when data is implicitly converted.

    This warning is issued when input data is automatically converted
    to a different type or format, which may lead to unexpected behavior
    or performance issues.
    """

    pass


class EfficiencyWarning(UserWarning):
    """Warning issued when an inefficient operation is detected.

    This warning is issued when an operation could be performed more
    efficiently with a different configuration or data format.
    """

    pass


def check_is_fitted(
    estimator: Any,
    attributes: Optional[list[str]] = None,
    msg: Optional[str] = None,
    all_or_any: str = "all",
) -> None:
    """Check if the estimator is fitted by verifying the presence of fitted attributes.

    Parameters
    ----------
    estimator : object
        Estimator instance to check.
    attributes : list of str or None, default=None
        Attributes to check. If None, checks for attributes ending with underscore.
    msg : str or None, default=None
        Custom error message. If None, a default message is used.
    all_or_any : {'all', 'any'}, default='all'
        Whether all or any of the attributes must be present.

    Raises
    ------
    NotFittedError
        If the estimator is not fitted.

    Examples
    --------
    >>> from eclipsera.ml.linear import LinearRegression
    >>> from eclipsera.core.exceptions import check_is_fitted
    >>> model = LinearRegression()
    >>> try:
    ...     check_is_fitted(model, ['coef_', 'intercept_'])
    ... except NotFittedError:
    ...     print("Not fitted")
    Not fitted
    """
    if msg is None:
        msg = (
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )

    if attributes is None:
        # Check for any attribute ending with underscore (sklearn convention)
        fitted_attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]
        if not fitted_attrs:
            raise NotFittedError(msg)
    else:
        if all_or_any == "all":
            if not all(hasattr(estimator, attr) for attr in attributes):
                raise NotFittedError(msg)
        elif all_or_any == "any":
            if not any(hasattr(estimator, attr) for attr in attributes):
                raise NotFittedError(msg)
        else:
            raise ValueError("all_or_any must be 'all' or 'any'")
