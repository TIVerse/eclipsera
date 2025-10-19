"""Matrix decomposition and dimensionality reduction."""
from ._pca import PCA
from ._truncated_svd import TruncatedSVD
from ._nmf import NMF

__all__ = [
    "PCA",
    "TruncatedSVD",
    "NMF",
]
