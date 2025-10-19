Quick Start Guide
=================

This guide will get you up and running with Eclipsera in minutes.

Installation
------------

Install Eclipsera using pip:

.. code-block:: bash

    pip install eclipsera

Basic Classification Example
-----------------------------

Here's a simple classification example:

.. code-block:: python

    import numpy as np
    from eclipsera.ml import RandomForestClassifier
    from eclipsera.model_selection import train_test_split

    # Generate sample data
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")

    # Make predictions
    predictions = clf.predict(X_test)

AutoML - Automatic Model Selection
-----------------------------------

Let Eclipsera automatically find the best algorithm for your data:

.. code-block:: python

    from eclipsera.automl import AutoClassifier

    # Automatically try multiple algorithms
    auto_clf = AutoClassifier(cv=5, verbose=1)
    auto_clf.fit(X_train, y_train)

    # AutoClassifier: Evaluating 6 algorithms...
    #   Trying logistic_regression... Score: 0.8524
    #   Trying random_forest... Score: 0.9123
    #   ...
    # Best algorithm: random_forest (score: 0.9123)

    print(f"Best algorithm: {auto_clf.best_algorithm_}")
    print(f"Best score: {auto_clf.best_score_:.4f}")

    # Use like any other classifier
    y_pred = auto_clf.predict(X_test)

Explain Model Predictions
--------------------------

Understand which features are most important:

.. code-block:: python

    from eclipsera.explainability import permutation_importance
    from eclipsera.ml import RandomForestClassifier

    # Train a model
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    # Compute feature importance
    result = permutation_importance(
        clf, X_test, y_test, 
        n_repeats=10, 
        random_state=42
    )

    # Display results
    for i in range(X.shape[1]):
        print(f"Feature {i}: {result['importances_mean'][i]:.4f} "
              f"± {result['importances_std'][i]:.4f}")

Complete Pipeline
-----------------

Build an end-to-end ML pipeline:

.. code-block:: python

    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    from eclipsera.feature_selection import SelectKBest
    from eclipsera.decomposition import PCA
    from eclipsera.ml import LogisticRegression

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(k=8)),
        ('pca', PCA(n_components=5)),
        ('clf', LogisticRegression())
    ])

    # Train pipeline
    pipe.fit(X_train, y_train)

    # Evaluate
    score = pipe.score(X_test, y_test)
    print(f"Pipeline accuracy: {score:.3f}")

Clustering
----------

Group similar data points together:

.. code-block:: python

    from eclipsera.cluster import KMeans

    # Cluster data into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    # Get cluster centers
    centers = kmeans.cluster_centers_

Dimensionality Reduction
-------------------------

Reduce data dimensions while preserving information:

.. code-block:: python

    from eclipsera.decomposition import PCA

    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Explained variance
    print(f"Explained variance: {pca.explained_variance_ratio_}")

Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations
* Explore :doc:`tutorials/classification` for more examples
* Check the :doc:`api/ml` for available algorithms
* Learn about :doc:`tutorials/automl` for automatic optimization
* Understand your models with :doc:`tutorials/explainability`
