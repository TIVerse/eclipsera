AutoML Tutorial
===============

This tutorial demonstrates how to use Eclipsera's AutoML capabilities for automatic algorithm selection.

What is AutoML?
---------------

AutoML (Automated Machine Learning) automatically selects and tunes the best algorithm for your data, 
saving time and requiring no expert knowledge of algorithms.

AutoClassifier
--------------

Automatically select the best classification algorithm:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from eclipsera.automl import AutoClassifier
    from eclipsera.model_selection import train_test_split

    # Generate sample data
    X = np.random.randn(300, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Auto-select best classifier
    auto_clf = AutoClassifier(cv=5, verbose=1)
    auto_clf.fit(X_train, y_train)

    # Output:
    # AutoClassifier: Evaluating 6 algorithms...
    #   Trying logistic_regression... Score: 0.8524
    #   Trying random_forest... Score: 0.9123
    #   Trying gradient_boosting... Score: 0.9087
    #   Trying knn... Score: 0.8891
    #   Trying decision_tree... Score: 0.8456
    #   Trying naive_bayes... Score: 0.8234
    # 
    # Best algorithm: random_forest (score: 0.9123)

    # Use like any classifier
    y_pred = auto_clf.predict(X_test)
    accuracy = auto_clf.score(X_test, y_test)

    print(f"Best algorithm: {auto_clf.best_algorithm_}")
    print(f"CV score: {auto_clf.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Access all scores
    print("All algorithm scores:")
    for name, score in auto_clf.scores_.items():
        print(f"  {name}: {score:.4f}")

    # Access the best fitted model
    best_model = auto_clf.best_estimator_
    print(f"Best model: {best_model}")

    # Get feature importances (if available)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        print(f"Feature importances: {importances}")

Selecting Specific Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose which algorithms to try:

.. code-block:: python

    # Only try specific algorithms
    auto_clf = AutoClassifier(
        algorithms=['logistic_regression', 'random_forest', 'gradient_boosting'],
        cv=5,
        verbose=1
    )
    auto_clf.fit(X_train, y_train)

    # Now only 3 algorithms will be evaluated

AutoRegressor
-------------

Automatically select the best regression algorithm:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from eclipsera.automl import AutoRegressor

    # Generate regression data
    X = np.random.randn(300, 15)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(300) * 0.1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Auto-select best regressor
    auto_reg = AutoRegressor(cv=5, scoring='r2', verbose=1)
    auto_reg.fit(X_train, y_train)

    # Output:
    # AutoRegressor: Evaluating 7 algorithms...
    #   Trying linear_regression... Score: 0.9456
    #   Trying ridge... Score: 0.9458
    #   Trying lasso... Score: 0.9312
    #   Trying random_forest... Score: 0.9234
    #   ...
    # 
    # Best algorithm: ridge (score: 0.9458)

    # Make predictions
    y_pred = auto_reg.predict(X_test)
    r2_score = auto_reg.score(X_test, y_test)

    print(f"Best algorithm: {auto_reg.best_algorithm_}")
    print(f"CV R² score: {auto_reg.best_score_:.4f}")
    print(f"Test R² score: {r2_score:.4f}")

Different Scoring Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use Mean Squared Error as metric
    auto_reg = AutoRegressor(
        cv=5,
        scoring='neg_mean_squared_error',  # Negative because higher is better
        verbose=1
    )
    auto_reg.fit(X_train, y_train)

Real-World Example: Iris Classification
----------------------------------------

.. code-block:: python

    from sklearn.datasets import load_iris
    from eclipsera.automl import AutoClassifier
    from eclipsera.model_selection import train_test_split

    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Auto-select best classifier
    auto_clf = AutoClassifier(cv=5, verbose=1, random_state=42)
    auto_clf.fit(X_train, y_train)

    # Evaluate
    train_acc = auto_clf.score(X_train, y_train)
    test_acc = auto_clf.score(X_test, y_test)

    print(f"\\nResults:")
    print(f"Best algorithm: {auto_clf.best_algorithm_}")
    print(f"Cross-validation score: {auto_clf.best_score_:.4f}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Check for overfitting
    if train_acc - test_acc > 0.1:
        print("Warning: Possible overfitting detected")

Integration with Pipelines
---------------------------

AutoML can be used in pipelines:

.. code-block:: python

    from eclipsera.pipeline import Pipeline
    from eclipsera.preprocessing import StandardScaler
    from eclipsera.automl import AutoClassifier

    # Create pipeline with AutoML
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('auto_clf', AutoClassifier(cv=3, verbose=0))
    ])

    # Train pipeline
    pipe.fit(X_train, y_train)

    # The pipeline will automatically select the best algorithm
    # after scaling the data

    print(f"Best algorithm in pipeline: {pipe.named_steps['auto_clf'].best_algorithm_}")

Tips and Best Practices
-----------------------

1. **Start with verbose=1** to see what algorithms are being tried
2. **Use cross-validation** (cv=5 or cv=10) for reliable estimates
3. **Set random_state** for reproducibility
4. **Try specific algorithms** if you know some work better for your problem
5. **Check for overfitting** by comparing training and test scores
6. **Use AutoML for baselines** then manually tune the best algorithm
7. **Scale your data** before AutoML for best results

Limitations
-----------

* AutoML tries each algorithm with default parameters
* For best results, manual hyperparameter tuning may be needed after selection
* Evaluation time grows with number of samples and algorithms
* Some algorithms may fail on certain data types

Next Steps
----------

* Learn about :doc:`explainability` to understand the chosen model
* Try :doc:`pipelines` to integrate AutoML into workflows
* Explore manual :doc:`classification` for fine-tuning
* Read about :doc:`../api/automl` for full API reference
