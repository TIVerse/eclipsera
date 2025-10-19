"""Eclipsera: A Modern Machine Learning Framework.

Eclipsera is a comprehensive machine learning framework with 68 algorithms
spanning classical ML, clustering, dimensionality reduction, manifold learning,
AutoML, and explainability.
"""
from . import (
    automl,
    cli,
    cluster,
    core,
    decomposition,
    explainability,
    feature_selection,
    manifold,
    ml,
    model_selection,
    pipeline,
    preprocessing,
)
from .__version__ import (
    __author__,
    __copyright__,
    __email__,
    __license__,
    __version__,
)


def show_versions() -> None:
    """Print version information for Eclipsera and dependencies.

    This is useful for debugging and reporting issues.

    Examples
    --------
    >>> import eclipsera
    >>> eclipsera.show_versions()  # doctest: +SKIP
    """
    import platform
    import sys

    print(f"Eclipsera version: {__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("\nDependencies:")

    deps = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "joblib",
        "numba",
        "matplotlib",
    ]

    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {dep}: {version}")
        except ImportError:
            print(f"  {dep}: not installed")


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "show_versions",
    # Core
    "BaseEstimator",
    "BaseClassifier",
    "BaseRegressor",
    "BaseTransformer",
    "clone",
    # Metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "r2_score",
    "mean_squared_error",
    "mean_absolute_error",
    # Validation
    "check_array",
    "check_X_y",
    # Modules
    "core",
    "ml",
    "preprocessing",
]
