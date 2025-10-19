AutoML API Reference
====================

The ``eclipsera.automl`` module provides automatic algorithm selection and optimization.

AutoClassifier
--------------

.. autoclass:: eclipsera.automl.AutoClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   Automatic classifier selection and optimization.

   **Parameters:**

   - **scoring** (*str, default='accuracy'*) - Metric to use for evaluation
   - **cv** (*int, default=5*) - Number of cross-validation folds
   - **algorithms** (*list of str, default=None*) - List of algorithm names to try
   - **verbose** (*int, default=1*) - Verbosity level
   - **random_state** (*int, default=None*) - Random state for reproducibility

   **Attributes:**

   - **best_estimator_** - The best performing estimator
   - **best_score_** (*float*) - Score of the best estimator
   - **best_algorithm_** (*str*) - Name of the best algorithm
   - **scores_** (*dict*) - Scores for all tried algorithms

   **Methods:**

   .. method:: fit(X, y)

      Find the best classifier for the data.

      :param X: Training data
      :type X: array-like of shape (n_samples, n_features)
      :param y: Target values
      :type y: array-like of shape (n_samples,)
      :return: self
      :rtype: AutoClassifier

   .. method:: predict(X)

      Predict using the best estimator.

      :param X: Samples to predict
      :type X: array-like of shape (n_samples, n_features)
      :return: Predicted class labels
      :rtype: ndarray of shape (n_samples,)

   .. method:: predict_proba(X)

      Predict class probabilities using the best estimator.

      :param X: Samples to predict
      :type X: array-like of shape (n_samples, n_features)
      :return: Class probabilities
      :rtype: ndarray of shape (n_samples, n_classes)

   .. method:: score(X, y)

      Score using the best estimator.

      :param X: Test samples
      :type X: array-like of shape (n_samples, n_features)
      :param y: True labels
      :type y: array-like of shape (n_samples,)
      :return: Score of the best estimator
      :rtype: float

AutoRegressor
-------------

.. autoclass:: eclipsera.automl.AutoRegressor
   :members:
   :undoc-members:
   :show-inheritance:

   Automatic regressor selection and optimization.

   **Parameters:**

   - **scoring** (*str, default='r2'*) - Metric to use for evaluation
   - **cv** (*int, default=5*) - Number of cross-validation folds
   - **algorithms** (*list of str, default=None*) - List of algorithm names to try
   - **verbose** (*int, default=1*) - Verbosity level
   - **random_state** (*int, default=None*) - Random state for reproducibility

   **Attributes:**

   - **best_estimator_** - The best performing estimator
   - **best_score_** (*float*) - Score of the best estimator
   - **best_algorithm_** (*str*) - Name of the best algorithm
   - **scores_** (*dict*) - Scores for all tried algorithms

   **Methods:**

   .. method:: fit(X, y)

      Find the best regressor for the data.

      :param X: Training data
      :type X: array-like of shape (n_samples, n_features)
      :param y: Target values
      :type y: array-like of shape (n_samples,)
      :return: self
      :rtype: AutoRegressor

   .. method:: predict(X)

      Predict using the best estimator.

      :param X: Samples to predict
      :type X: array-like of shape (n_samples, n_features)
      :return: Predicted values
      :rtype: ndarray of shape (n_samples,)

   .. method:: score(X, y)

      Score using the best estimator.

      :param X: Test samples
      :type X: array-like of shape (n_samples, n_features)
      :param y: True values
      :type y: array-like of shape (n_samples,)
      :return: Score of the best estimator
      :rtype: float

Examples
--------

Classification
~~~~~~~~~~~~~~

.. code-block:: python

    from eclipsera.automl import AutoClassifier
    
    auto_clf = AutoClassifier(cv=5, verbose=1)
    auto_clf.fit(X_train, y_train)
    
    print(f"Best: {auto_clf.best_algorithm_}")
    y_pred = auto_clf.predict(X_test)

Regression
~~~~~~~~~~

.. code-block:: python

    from eclipsera.automl import AutoRegressor
    
    auto_reg = AutoRegressor(cv=5, scoring='r2')
    auto_reg.fit(X_train, y_train)
    
    print(f"Best: {auto_reg.best_algorithm_}")
    y_pred = auto_reg.predict(X_test)
