"""Pipeline and composition tools."""
from ._pipeline import FeatureUnion, Pipeline, make_pipeline

__all__ = [
    "Pipeline",
    "FeatureUnion",
    "make_pipeline",
]
