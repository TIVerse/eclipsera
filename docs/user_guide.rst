User Guide
==========

This guide provides an overview of Eclipsera's capabilities and how to use them effectively.

Overview
--------

Eclipsera is a comprehensive machine learning framework with 68 algorithms organized into 12 modules:

1. **ml** - Supervised learning (28 algorithms)
2. **cluster** - Clustering (7 algorithms)
3. **decomposition** - Dimensionality reduction (3 algorithms)
4. **manifold** - Manifold learning (3 algorithms)
5. **feature_selection** - Feature selection (3 tools)
6. **preprocessing** - Data preprocessing (10 tools)
7. **model_selection** - Cross-validation and hyperparameter tuning (8 utilities)
8. **pipeline** - Workflow composition (3 tools)
9. **automl** - Automatic model selection (2 tools)
10. **explainability** - Model interpretation (4 tools)
11. **core** - Base classes and utilities
12. **cli** - Command-line interface

Core Concepts
-------------

Estimators
~~~~~~~~~~

All machine learning algorithms in Eclipsera are estimators that follow the scikit-learn API:

.. code-block:: python

    from eclipsera.ml import RandomForestClassifier

    # Create estimator
    clf = RandomForestClassifier(n_estimators=100)

    # Fit on training data
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate
    score = clf.score(X_test, y_test)

Transformers
~~~~~~~~~~~~

Transformers modify data and follow the fit/transform pattern:

.. code-block:: python

    from eclipsera.preprocessing import StandardScaler

    # Create transformer
    scaler = StandardScaler()

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data (using training statistics)
    X_test_scaled = scaler.transform(X_test)

Pipelines
~~~~~~~~~

Pipelines chain multiple steps into a single estimator:

.. code-block:: python

    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    from eclipsera.ml import LogisticRegression

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    # Train entire pipeline
    pipe.fit(X_train, y_train)

    # Predict with pipeline
    y_pred = pipe.predict(X_test)

Supervised Learning
-------------------

Classification
~~~~~~~~~~~~~~

Eclipsera provides 14 classification algorithms:

.. code-block:: python

    from eclipsera.ml import (
        LogisticRegression,
        RandomForestClassifier,
        GradientBoostingClassifier,
        SVC,
        KNeighborsClassifier,
        MLPClassifier
    )

    # Train any classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Binary or multiclass classification
    y_pred = clf.predict(X_test)

    # Get probabilities (if supported)
    if hasattr(clf, 'predict_proba'):
        probas = clf.predict_proba(X_test)

Regression
~~~~~~~~~~

Eclipsera provides 14 regression algorithms:

.. code-block:: python

    from eclipsera.ml import (
        LinearRegression,
        Ridge,
        Lasso,
        RandomForestRegressor,
        GradientBoostingRegressor
    )

    # Train regressor
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)

    # Predict continuous values
    y_pred = reg.predict(X_test)

    # Evaluate with R² score
    score = reg.score(X_test, y_test)

Unsupervised Learning
---------------------

Clustering
~~~~~~~~~~

Group similar data points:

.. code-block:: python

    from eclipsera.cluster import (
        KMeans,
        DBSCAN,
        AgglomerativeClustering,
        SpectralClustering,
        GaussianMixture
    )

    # K-Means clustering
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(X)

    # DBSCAN (density-based)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)

    # Gaussian Mixture (probabilistic)
    gmm = GaussianMixture(n_components=3)
    labels = gmm.fit_predict(X)
    probabilities = gmm.predict_proba(X)

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~

Reduce feature dimensions:

.. code-block:: python

    from eclipsera.decomposition import PCA, NMF

    # PCA for linear dimensionality reduction
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)

    # Explained variance
    print(pca.explained_variance_ratio_)

    # Inverse transform
    X_reconstructed = pca.inverse_transform(X_reduced)

Manifold Learning
~~~~~~~~~~~~~~~~~

Non-linear dimensionality reduction:

.. code-block:: python

    from eclipsera.manifold import TSNE, Isomap, LocallyLinearEmbedding

    # t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(X)

    # Isomap preserves geodesic distances
    isomap = Isomap(n_components=2, n_neighbors=5)
    X_embedded = isomap.fit_transform(X)

Feature Engineering
-------------------

Feature Selection
~~~~~~~~~~~~~~~~~

Select the most important features:

.. code-block:: python

    from eclipsera.feature_selection import (
        SelectKBest,
        RFE,
        VarianceThreshold
    )

    # Select top K features
    selector = SelectKBest(k=10)
    X_selected = selector.fit_transform(X, y)

    # Recursive feature elimination
    from eclipsera.ml import LogisticRegression
    rfe = RFE(LogisticRegression(), n_features_to_select=10)
    X_selected = rfe.fit_transform(X, y)

Data Preprocessing
~~~~~~~~~~~~~~~~~~

Prepare data for machine learning:

.. code-block:: python

    from eclipsera.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        SimpleImputer,
        LabelEncoder,
        OneHotEncoder
    )

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Missing value imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Encoding categorical variables
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X_categorical)

AutoML
------

Automatic Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let Eclipsera choose the best algorithm:

.. code-block:: python

    from eclipsera.automl import AutoClassifier, AutoRegressor

    # Classification
    auto_clf = AutoClassifier(cv=5, verbose=1)
    auto_clf.fit(X_train, y_train)

    print(f"Best: {auto_clf.best_algorithm_}")
    print(f"Score: {auto_clf.best_score_:.4f}")

    # Regression
    auto_reg = AutoRegressor(cv=5, scoring='r2', verbose=1)
    auto_reg.fit(X_train, y_train)

Model Explainability
--------------------

Understand Model Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.explainability import (
        permutation_importance,
        partial_dependence,
        get_feature_importance
    )

    # Permutation importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10)
    print(result['importances_mean'])

    # Partial dependence
    pd_result = partial_dependence(clf, X, features=[0, 1, 2])

    # Direct feature importance (for tree-based models)
    imp = get_feature_importance(clf, feature_names=['age', 'income', 'score'])

Model Evaluation
----------------

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.model_selection import cross_val_score, KFold

    # Simple cross-validation
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Custom cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.model_selection import GridSearchCV, RandomizedSearchCV

    # Grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best score: {grid.best_score_:.4f}")

Best Practices
--------------

1. **Always split your data** into training and test sets
2. **Scale your features** when using distance-based algorithms
3. **Use cross-validation** for reliable performance estimates
4. **Start simple** with linear models before trying complex ones
5. **Check for overfitting** by comparing training and test scores
6. **Use pipelines** to prevent data leakage
7. **Set random_state** for reproducibility
8. **Explain your models** to build trust and understanding

Next Steps
----------

* Explore detailed :doc:`tutorials/classification`
* Learn about :doc:`tutorials/automl`
* Master :doc:`tutorials/pipelines`
* Understand :doc:`tutorials/explainability`
* Check the :doc:`api/ml` reference
