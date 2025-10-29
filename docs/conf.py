"""Sphinx configuration for Eclipsera documentation."""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))

# Import version
from eclipsera.__version__ import __version__

# Project information
project = "Eclipsera"
copyright = f"{datetime.now().year}, Tonmoy Infrastructure & Vision"
author = "Eshan Roy"
version = __version__
release = __version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "nbsphinx",
]

# Nitpicky mode for strict cross-reference checking
nitpicky = True
nitpick_ignore = [
    ("py:class", "best_estimator"),
    ("py:class", "best_score"),
    ("py:class", "best_algorithm"),
    ("py:class", "scores"),
    ("py:class", "cluster_centers"),
    ("py:class", "labels"),
    ("py:class", "inertia"),
    ("py:class", "n_iter"),
    ("py:class", "core_sample_indices"),
    ("py:class", "components"),
    ("py:class", "n_features_in"),
    ("py:class", "n_clusters"),
    ("py:class", "mean"),
    ("py:class", "scale"),
    ("py:class", "var"),
    ("py:class", "min"),
    ("py:class", "data_min"),
    ("py:class", "data_max"),
    ("py:class", "data_range"),
    ("py:class", "max_abs"),
    ("py:class", "center"),
    ("py:class", "classes"),
    ("py:class", "categories"),
    ("py:class", "statistics"),
    ("py:class", "x"),
    ("py:class", "cv_results"),
    # Numpy type references
    ("py:class", "array-like"),
    ("py:class", "shape"),
    ("py:class", "n_samples"),
    ("py:class", "n_features"),
    ("py:class", "ndarray"),
]

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"

# HTML output
html_theme = "furo"
html_title = f"Eclipsera {version}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "no-index": True,  # Avoid duplicate warnings
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Todo extension
todo_include_todos = True

# nbsphinx settings
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# pygments style
pygments_style = "sphinx"
