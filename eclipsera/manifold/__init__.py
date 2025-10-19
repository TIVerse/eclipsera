"""Manifold learning algorithms."""
from ._tsne import TSNE
from ._isomap import Isomap
from ._lle import LocallyLinearEmbedding

__all__ = [
    "TSNE",
    "Isomap",
    "LocallyLinearEmbedding",
]
