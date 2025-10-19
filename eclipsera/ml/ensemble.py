"""Ensemble methods including Random Forest and Gradient Boosting."""
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from ..core.base import BaseClassifier, BaseRegressor, clone
from ..core.exceptions import NotFittedError
from ..core.utils import check_random_state
from ..core.validation import check_array, check_X_y
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseClassifier):
    """A random forest classifier.
    
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2'}, int, float or None, default='sqrt'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel. -1 means using all processors.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping and feature sampling.
        
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.ensemble import RandomForestClassifier
    >>> X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    >>> y = np.array([0, 1, 1, 0])
    >>> clf = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> clf.fit(X, y)
    RandomForestClassifier(n_estimators=10, random_state=42)
    >>> clf.predict([[0.5, 0.5]])
    array([0])
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Build a forest of trees from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : RandomForestClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        random_state = check_random_state(self.random_state)
        
        # Determine max_features
        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * self.n_features_in_))
        else:
            max_features = self.n_features_in_
        
        # Create base estimators
        seeds = random_state.randint(0, 10000, size=self.n_estimators)
        
        def _fit_tree(seed):
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=seed,
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = np.random.RandomState(seed).choice(
                    n_samples, size=n_samples, replace=True
                )
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)
            
            return tree
        
        # Fit trees in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(seed) for seed in seeds
        )
        
        # Calculate feature importances
        self.feature_importances_ = np.mean([
            tree.feature_importances_ for tree in self.estimators_
        ], axis=0)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = check_array(X)
        
        if not hasattr(self, 'estimators_'):
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        
        # Average predictions from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return np.mean(all_proba, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RandomForestRegressor(BaseRegressor):
    """A random forest regressor.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'mse', 'mae'}, default='mse'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2'}, int, float or None, default='sqrt'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        
    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Build a forest of trees from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : RandomForestRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        
        random_state = check_random_state(self.random_state)
        
        # Determine max_features
        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * self.n_features_in_))
        else:
            max_features = self.n_features_in_
        
        seeds = random_state.randint(0, 10000, size=self.n_estimators)
        
        def _fit_tree(seed):
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=seed,
            )
            
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = np.random.RandomState(seed).choice(
                    n_samples, size=n_samples, replace=True
                )
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)
            
            return tree
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(seed) for seed in seeds
        )
        
        self.feature_importances_ = np.mean([
            tree.feature_importances_ for tree in self.estimators_
        ], axis=0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'estimators_'):
            raise NotFittedError("This RandomForestRegressor instance is not fitted yet.")
        
        # Average predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(all_predictions, axis=0)


class GradientBoostingClassifier(BaseClassifier):
    """Gradient Boosting for classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base learners.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        
    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """Fit the gradient boosting model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : GradientBoostingClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        random_state = check_random_state(self.random_state)
        
        # Convert labels to indices
        y_encoded = np.searchsorted(self.classes_, y)
        
        if self.n_classes_ == 2:
            # Binary classification
            self.estimators_ = []
            
            # Initialize with log odds
            pos_count = np.sum(y_encoded == 1)
            neg_count = len(y_encoded) - pos_count
            init_pred = np.log(pos_count / (neg_count + 1e-10))
            self.init_pred_ = init_pred
            
            # Current predictions (in log odds space)
            f = np.full(len(y_encoded), init_pred)
            
            for i in range(self.n_estimators):
                # Compute gradients (negative gradient for gradient descent)
                p = 1 / (1 + np.exp(-f))
                residuals = y_encoded - p
                
                # Subsample
                if self.subsample < 1.0:
                    n_samples = int(self.subsample * len(X))
                    indices = random_state.choice(len(X), size=n_samples, replace=False)
                    X_sub = X[indices]
                    residuals_sub = residuals[indices]
                else:
                    X_sub = X
                    residuals_sub = residuals
                
                # Fit tree to residuals
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=random_state.randint(0, 10000),
                )
                tree.fit(X_sub, residuals_sub)
                
                # Update predictions
                update = tree.predict(X)
                f += self.learning_rate * update
                
                self.estimators_.append(tree)
        else:
            # Multiclass: one-vs-rest
            self.estimators_ = []
            self.init_pred_ = np.zeros(self.n_classes_)
            
            for class_idx in range(self.n_classes_):
                class_estimators = []
                y_binary = (y_encoded == class_idx).astype(float)
                
                # Initialize
                pos_count = np.sum(y_binary)
                neg_count = len(y_binary) - pos_count
                init_pred = np.log(pos_count / (neg_count + 1e-10))
                self.init_pred_[class_idx] = init_pred
                
                f = np.full(len(y_binary), init_pred)
                
                for i in range(self.n_estimators):
                    p = 1 / (1 + np.exp(-f))
                    residuals = y_binary - p
                    
                    if self.subsample < 1.0:
                        n_samples = int(self.subsample * len(X))
                        indices = random_state.choice(len(X), size=n_samples, replace=False)
                        X_sub = X[indices]
                        residuals_sub = residuals[indices]
                    else:
                        X_sub = X
                        residuals_sub = residuals
                    
                    tree = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=random_state.randint(0, 10000),
                    )
                    tree.fit(X_sub, residuals_sub)
                    
                    update = tree.predict(X)
                    f += self.learning_rate * update
                    
                    class_estimators.append(tree)
                
                self.estimators_.append(class_estimators)
        
        return self
    
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw predictions (before sigmoid/softmax)."""
        X = check_array(X)
        
        if self.n_classes_ == 2:
            # Binary
            predictions = np.full(len(X), self.init_pred_)
            for tree in self.estimators_:
                predictions += self.learning_rate * tree.predict(X)
            return predictions
        else:
            # Multiclass
            predictions = np.tile(self.init_pred_, (len(X), 1))
            for class_idx in range(self.n_classes_):
                for tree in self.estimators_[class_idx]:
                    predictions[:, class_idx] += self.learning_rate * tree.predict(X)
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities.
        """
        if not hasattr(self, 'estimators_'):
            raise NotFittedError("This GradientBoostingClassifier instance is not fitted yet.")
        
        raw_predictions = self._raw_predict(X)
        
        if self.n_classes_ == 2:
            proba_class1 = 1 / (1 + np.exp(-raw_predictions))
            return np.column_stack([1 - proba_class1, proba_class1])
        else:
            # Softmax
            exp_pred = np.exp(raw_predictions - np.max(raw_predictions, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class GradientBoostingRegressor(BaseRegressor):
    """Gradient Boosting for regression.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """Fit the gradient boosting model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : GradientBoostingRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        random_state = check_random_state(self.random_state)
        
        # Initialize with mean
        self.init_pred_ = np.mean(y)
        
        # Current predictions
        f = np.full(len(y), self.init_pred_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - f
            
            # Subsample
            if self.subsample < 1.0:
                n_samples = int(self.subsample * len(X))
                indices = random_state.choice(len(X), size=n_samples, replace=False)
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=random_state.randint(0, 10000),
            )
            tree.fit(X_sub, residuals_sub)
            
            # Update predictions
            update = tree.predict(X)
            f += self.learning_rate * update
            
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X)
        
        if not hasattr(self, 'estimators_'):
            raise NotFittedError("This GradientBoostingRegressor instance is not fitted yet.")
        
        predictions = np.full(len(X), self.init_pred_)
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
