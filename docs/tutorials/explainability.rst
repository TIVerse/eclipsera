Model Explainability Tutorial
=============================

This tutorial shows how to explain and interpret machine learning model predictions.

Why Explainability Matters
---------------------------

Understanding why a model makes certain predictions is crucial for:

* Building trust in model decisions
* Debugging model behavior
* Regulatory compliance (GDPR, etc.)
* Feature engineering insights
* Communicating results to stakeholders

Eclipsera provides model-agnostic explainability tools that work with any estimator.

Permutation Importance
----------------------

Measure feature importance by permuting feature values:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from eclipsera.ml import RandomForestClassifier
    from eclipsera.explainability import permutation_importance
    from eclipsera.model_selection import train_test_split

    # Generate data
    X = np.random.randn(200, 10)
    # Make first 3 features informative
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    # Compute permutation importance
    result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=42
    )

    # Display results
    print("Feature Importances:")
    for i in range(X.shape[1]):
        print(f"Feature {i}: {result['importances_mean'][i]:.4f} "
              f"± {result['importances_std'][i]:.4f}")

    # Output shows features 0, 1, 2 have highest importance

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get full importance matrix
    importances = result['importances']  # Shape: (n_features, n_repeats)

    # Mean importance across repeats
    mean_imp = result['importances_mean']

    # Standard deviation (statistical significance)
    std_imp = result['importances_std']

    # Sort features by importance
    indices = np.argsort(mean_imp)[::-1]

    print("\\nTop 5 most important features:")
    for i in indices[:5]:
        print(f"  Feature {i}: {mean_imp[i]:.4f} ± {std_imp[i]:.4f}")

With Feature Names
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # More readable with feature names
    feature_names = ['age', 'income', 'credit_score', 'balance', 
                     'tenure', 'products', 'active', 'salary',
                     'region', 'gender']

    result = permutation_importance(clf, X_test, y_test, n_repeats=10)

    print("\\nFeature Importance:")
    for i, name in enumerate(feature_names):
        print(f"{name:15s}: {result['importances_mean'][i]:.4f}")

Partial Dependence
------------------

Visualize how features affect predictions:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from eclipsera.explainability import partial_dependence

    # Compute partial dependence for features 0, 1, 2
    pd_result = partial_dependence(
        clf, X_test,
        features=[0, 1, 2],
        grid_resolution=50
    )

    # Access results
    for i, feature_idx in enumerate([0, 1, 2]):
        grid_values = pd_result['values'][i]
        predictions = pd_result['predictions'][i]

        print(f"\\nFeature {feature_idx}:")
        print(f"  Grid range: [{grid_values.min():.2f}, {grid_values.max():.2f}]")
        print(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

Plotting Partial Dependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.explainability import plot_partial_dependence

    # Plot for multiple features
    result = plot_partial_dependence(
        clf, X_test,
        features=[0, 1, 2],
        feature_names=['age', 'income', 'credit_score'],
        grid_resolution=100
    )

    # Creates a matplotlib figure with 3 subplots
    # showing how each feature affects predictions

Custom Grid Resolution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Higher resolution for smoother curves
    pd_result = partial_dependence(
        clf, X_test,
        features=[0],
        grid_resolution=200  # More points = smoother curve
    )

    # Custom percentiles (default is 5th to 95th)
    pd_result = partial_dependence(
        clf, X_test,
        features=[0],
        percentiles=(0.1, 0.9)  # Focus on 10th to 90th percentile
    )

Feature Importance Extraction
------------------------------

Get built-in feature importances from models:

Tree-Based Models
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.explainability import get_feature_importance
    from eclipsera.ml import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Extract feature importances
    result = get_feature_importance(
        clf,
        feature_names=['age', 'income', 'credit_score', 'balance']
    )

    print("Feature Importance (from trees):")
    for i in result['sorted_idx']:
        name = result['feature_names'][i]
        importance = result['importances'][i]
        print(f"  {name:15s}: {importance:.4f}")

Linear Models
~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.ml import LogisticRegression

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Extract coefficient importances
    result = get_feature_importance(clf, feature_names=feature_names)

    print("\\nFeature Importance (from coefficients):")
    for i in result['sorted_idx']:
        name = result['feature_names'][i]
        importance = result['importances'][i]
        print(f"  {name:15s}: {importance:.4f}")

Real-World Example: Credit Scoring
-----------------------------------

.. code-block:: python

    import numpy as np
    from eclipsera.ml import GradientBoostingClassifier
    from eclipsera.explainability import (
        permutation_importance,
        partial_dependence,
        get_feature_importance
    )

    # Simulate credit scoring data
    np.random.seed(42)
    n_samples = 1000

    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    debt_ratio = np.random.uniform(0, 1, n_samples)

    X = np.column_stack([age, income, credit_score, debt_ratio])

    # Credit approval based on features
    y = ((credit_score > 650) & 
         (income > 40000) & 
         (debt_ratio < 0.5)).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1))
    y[noise_idx] = 1 - y[noise_idx]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print(f"Model Accuracy: {clf.score(X_test, y_test):.4f}")

    # 1. Built-in feature importance
    feature_names = ['age', 'income', 'credit_score', 'debt_ratio']
    imp = get_feature_importance(clf, feature_names=feature_names)

    print("\\nBuilt-in Feature Importance:")
    for i in imp['sorted_idx']:
        print(f"  {imp['feature_names'][i]:15s}: {imp['importances'][i]:.4f}")

    # 2. Permutation importance
    perm_imp = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42
    )

    print("\\nPermutation Importance:")
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: {perm_imp['importances_mean'][i]:.4f} "
              f"± {perm_imp['importances_std'][i]:.4f}")

    # 3. Partial dependence
    print("\\nPartial Dependence Analysis:")
    for i, name in enumerate(feature_names):
        pd_result = partial_dependence(clf, X_test, features=[i])
        values = pd_result['values'][0]
        predictions = pd_result['predictions'][0]

        print(f"  {name:15s}: range [{values.min():.1f}, {values.max():.1f}], "
              f"effect [{predictions.min():.3f}, {predictions.max():.3f}]")

Comparing Different Methods
----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Get three types of importance
    feature_names = ['f0', 'f1', 'f2', 'f3', 'f4']

    # 1. Built-in (tree-based)
    builtin = get_feature_importance(clf, feature_names=feature_names)

    # 2. Permutation
    perm = permutation_importance(clf, X_test, y_test, n_repeats=10)

    # Plot comparison
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, builtin['importances'], width, label='Built-in')
    ax.bar(x + width/2, perm['importances_mean'], width, label='Permutation')

    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Comparison')
    ax.set_xticks(x)
    ax.set_xticks_labels(feature_names)
    ax.legend()
    plt.show()

Tips and Best Practices
-----------------------

1. **Use permutation importance** for model-agnostic analysis
2. **Increase n_repeats** (e.g., 30) for more reliable estimates
3. **Check standard deviation** to assess statistical significance
4. **Use partial dependence** to understand feature effects
5. **Compare multiple methods** for robust insights
6. **Scale features** before computing importance for fair comparison
7. **Use test data** for permutation importance to avoid overfitting
8. **Visualize results** for better communication

Limitations
-----------

* **Permutation importance** can be slow for large datasets
* **Partial dependence** assumes features are independent
* **Correlated features** may show misleading importances
* **Computational cost** grows with n_repeats and n_features

Next Steps
----------

* Apply explainability to your :doc:`automl` results
* Learn about :doc:`pipelines` for integrated workflows
* Explore :doc:`classification` for more model types
* Read the :doc:`../api/explainability` API reference
