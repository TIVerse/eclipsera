"""Decision tree algorithms for classification and regression."""
import warnings
from typing import Optional

import numpy as np

from ..core.base import BaseClassifier, BaseRegressor
from ..core.exceptions import NotFittedError
from ..core.utils import check_random_state
from ..core.validation import check_array, check_X_y


class _Node:
    """Internal node representation for decision tree."""
    
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        value: Optional[np.ndarray] = None,
        impurity: float = 0.0,
        n_samples: int = 0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples


class DecisionTreeClassifier(BaseClassifier):
    """A decision tree classifier.
    
    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    tree_ : _Node
        The underlying tree structure.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.tree import DecisionTreeClassifier
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = np.array([0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=1)
    >>> clf.fit(X, y)
    DecisionTreeClassifier(max_depth=1)
    >>> clf.predict([[2., 2.]])
    array([1])
    """
    
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build a decision tree classifier from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Convert y to class indices
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self.class_to_idx_[c] for c in y])
        
        self.random_state_ = check_random_state(self.random_state)
        
        # Determine max_features
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))
        else:
            self.max_features_ = self.n_features_in_
        
        # Build tree
        self.tree_ = self._grow_tree(X, y_encoded)
        
        # Calculate feature importances
        self.feature_importances_ = self._compute_feature_importances()
        
        return self
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Current impurity
        parent_impurity = self._calculate_impurity(y)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        # Randomly select features to consider
        features = self.random_state_.choice(
            n_features,
            size=min(self.max_features_, n_features),
            replace=False
        )
        
        for feature in features:
            # Get unique values and sort
            thresholds = np.unique(X[:, feature])
            
            # Try each unique value as threshold
            for threshold in thresholds:
                # Split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> _Node:
        """Recursively grow the decision tree."""
        n_samples = len(y)
        n_samples_per_class = np.bincount(y, minlength=self.n_classes_)
        predicted_class = np.argmax(n_samples_per_class)
        
        node = _Node(
            impurity=self._calculate_impurity(y),
            n_samples=n_samples,
        )
        
        # Check stopping criteria
        if depth >= self.max_depth if self.max_depth else False:
            node.value = n_samples_per_class / n_samples
            return node
        
        if n_samples < self.min_samples_split:
            node.value = n_samples_per_class / n_samples
            return node
        
        if len(np.unique(y)) == 1:
            node.value = n_samples_per_class / n_samples
            return node
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            node.value = n_samples_per_class / n_samples
            return node
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _predict_tree(self, x: np.ndarray, node: _Node) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        if node.value is not None:  # Is leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
    
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
        
        if not hasattr(self, 'tree_'):
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")
        
        probas = np.array([self._predict_tree(x, self.tree_) for x in X])
        return probas
    
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
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]
    
    def _compute_feature_importances(self) -> np.ndarray:
        """Compute feature importances."""
        importances = np.zeros(self.n_features_in_)
        
        def traverse(node: _Node, importance_accumulator: np.ndarray):
            if node.value is not None:  # Is leaf
                return
            
            # Importance is based on the reduction in impurity
            if node.left and node.right:
                n_total = node.n_samples
                n_left = node.left.n_samples
                n_right = node.right.n_samples
                
                importance = node.impurity - (
                    n_left / n_total * node.left.impurity +
                    n_right / n_total * node.right.impurity
                )
                importance_accumulator[node.feature] += importance * n_total
                
                traverse(node.left, importance_accumulator)
                traverse(node.right, importance_accumulator)
        
        traverse(self.tree_, importances)
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances


class DecisionTreeRegressor(BaseRegressor):
    """A decision tree regressor.
    
    Parameters
    ----------
    criterion : {'mse', 'mae'}, default='mse'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : int, float or None, default=None
        The number of features to consider when looking for the best split.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    tree_ : _Node
        The underlying tree structure.
    """
    
    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Build a decision tree regressor from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        self.random_state_ = check_random_state(self.random_state)
        
        # Determine max_features
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))
        else:
            self.max_features_ = self.n_features_in_
        
        # Build tree
        self.tree_ = self._grow_tree(X, y)
        
        # Calculate feature importances
        self.feature_importances_ = self._compute_feature_importances()
        
        return self
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0.0
        return np.var(y)
    
    def _mae(self, y: np.ndarray) -> float:
        """Calculate mean absolute error."""
        if len(y) == 0:
            return 0.0
        median = np.median(y)
        return np.mean(np.abs(y - median))
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == "mse":
            return self._mse(y)
        elif self.criterion == "mae":
            return self._mae(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        parent_impurity = self._calculate_impurity(y)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        # Randomly select features
        features = self.random_state_.choice(
            n_features,
            size=min(self.max_features_, n_features),
            replace=False
        )
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> _Node:
        """Recursively grow the decision tree."""
        n_samples = len(y)
        
        node = _Node(
            impurity=self._calculate_impurity(y),
            n_samples=n_samples,
        )
        
        # Check stopping criteria
        if depth >= self.max_depth if self.max_depth else False:
            node.value = np.mean(y)
            return node
        
        if n_samples < self.min_samples_split:
            node.value = np.mean(y)
            return node
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            node.value = np.mean(y)
            return node
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _predict_tree(self, x: np.ndarray, node: _Node) -> float:
        """Predict value for a single sample."""
        if node.value is not None:  # Is leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
    
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
        
        if not hasattr(self, 'tree_'):
            raise NotFittedError("This DecisionTreeRegressor instance is not fitted yet.")
        
        predictions = np.array([self._predict_tree(x, self.tree_) for x in X])
        return predictions
    
    def _compute_feature_importances(self) -> np.ndarray:
        """Compute feature importances."""
        importances = np.zeros(self.n_features_in_)
        
        def traverse(node: _Node, importance_accumulator: np.ndarray):
            if node.value is not None:  # Is leaf
                return
            
            if node.left and node.right:
                n_total = node.n_samples
                n_left = node.left.n_samples
                n_right = node.right.n_samples
                
                importance = node.impurity - (
                    n_left / n_total * node.left.impurity +
                    n_right / n_total * node.right.impurity
                )
                importance_accumulator[node.feature] += importance * n_total
                
                traverse(node.left, importance_accumulator)
                traverse(node.right, importance_accumulator)
        
        traverse(self.tree_, importances)
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances
