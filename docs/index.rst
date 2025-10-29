.. Eclipsera documentation master file

🌒 Eclipsera Documentation
==========================

**A Modern Machine Learning Framework for Python**

Eclipsera is a comprehensive machine learning framework built from scratch with 
68 algorithms spanning classical ML, clustering, dimensionality reduction, 
manifold learning, AutoML, and explainability.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   installation
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/automl
   api/cluster
   api/core
   api/decomposition
   api/explainability
   api/feature_selection
   api/manifold
   api/ml
   api/model_selection
   api/pipeline
   api/preprocessing

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/automl
   tutorials/classification
   tutorials/clustering
   tutorials/explainability
   tutorials/pipelines
   tutorials/regression

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing
   changelog

Quick Example
-------------

.. code-block:: python

    import numpy as np
    from eclipsera.ml import RandomForestClassifier
    from eclipsera.model_selection import train_test_split

    # Load data
    X = np.random.randn(150, 4)
    y = np.random.randint(0, 3, 150)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score:.3f}")

Key Features
------------

🤖 **AutoML**
   Automatic algorithm selection and optimization for classification and regression

🔍 **Explainability**
   Permutation importance, partial dependence, feature importance extraction

📊 **Supervised Learning**
   28 classification and regression algorithms including Random Forest, Gradient Boosting, SVM, Neural Networks

🎯 **Clustering**
   7 algorithms including K-Means, DBSCAN, Spectral, Hierarchical, Gaussian Mixture

📉 **Dimensionality Reduction**
   PCA, NMF, TruncatedSVD for feature extraction

🗺️ **Manifold Learning**
   t-SNE, Isomap, LLE for non-linear dimensionality reduction

⚙️ **Feature Selection**
   Variance threshold, univariate selection, recursive feature elimination

🔧 **Preprocessing**
   Scalers, imputers, encoders for data preparation

🔗 **Pipelines**
   Composable workflows for end-to-end ML

Installation
------------

.. code-block:: bash

    pip install eclipsera

**Requirements:**

- Python 3.11+
- NumPy
- SciPy
- matplotlib (optional, for plotting)

**From source:**

.. code-block:: bash

    git clone https://github.com/tiverse/eclipsera.git
    cd eclipsera
    pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
