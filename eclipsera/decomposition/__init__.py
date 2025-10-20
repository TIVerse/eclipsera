"""Matrix decomposition and dimensionality reduction."""

from ._nmf import NMF
from ._pca import PCA
from ._truncated_svd import TruncatedSVD

__all__ = [
    "PCA",
    "TruncatedSVD",
    "NMF",
]
