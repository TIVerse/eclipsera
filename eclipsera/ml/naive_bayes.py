"""Naive Bayes algorithms for classification."""
from typing import Optional

import numpy as np

from ..core.base import BaseClassifier
from ..core.exceptions import NotFittedError
from ..core.validation import check_array, check_X_y


class GaussianNB(BaseClassifier):
    """Gaussian Naive Bayes classifier.
    
    Implements the Gaussian Naive Bayes algorithm for classification.
    Assumes that the likelihood of the features is Gaussian.
    
    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    class_prior_ : ndarray of shape (n_classes,)
        Probability of each class.
    theta_ : ndarray of shape (n_classes, n_features)
        Mean of each feature per class.
    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.ml.naive_bayes import GaussianNB
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> clf = GaussianNB()
    >>> clf.fit(X, y)
    GaussianNB()
    >>> clf.predict([[-0.8, -1]])
    array([0])
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """Fit Gaussian Naive Bayes classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : GaussianNB
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Calculate class priors
        self.class_prior_ = np.array([
            np.mean(y == c) for c in self.classes_
        ])
        
        # Calculate mean and variance for each class
        self.theta_ = np.zeros((self.n_classes_, self.n_features_in_))
        self.var_ = np.zeros((self.n_classes_, self.n_features_in_))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            self.var_[idx, :] = np.var(X_c, axis=0)
        
        # Add smoothing to variance
        self.var_ += self.var_smoothing * np.var(X, axis=0).max()
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log-probability of samples for each class.
        """
        X = check_array(X)
        
        if not hasattr(self, 'theta_'):
            raise NotFittedError("This GaussianNB instance is not fitted yet.")
        
        # Log prior
        log_prior = np.log(self.class_prior_)
        
        # Log likelihood: sum of log Gaussian PDF
        log_likelihood = np.zeros((X.shape[0], self.n_classes_))
        
        for idx in range(self.n_classes_):
            # Gaussian log PDF: -0.5 * log(2π) - 0.5 * log(var) - 0.5 * ((x - mean)^2 / var)
            diff = X - self.theta_[idx]
            log_likelihood[:, idx] = (
                -0.5 * np.sum(np.log(2 * np.pi * self.var_[idx])) -
                0.5 * np.sum((diff ** 2) / self.var_[idx], axis=1)
            )
        
        # Log posterior = log prior + log likelihood
        log_posterior = log_prior + log_likelihood
        
        # Normalize (subtract log sum exp for numerical stability)
        log_prob_x = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        
        return log_posterior - log_prob_x
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of samples for each class.
        """
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on an array of test vectors X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class MultinomialNB(BaseClassifier):
    """Multinomial Naive Bayes classifier.
    
    Suitable for classification with discrete features (e.g., word counts
    for text classification).
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class.
    """
    
    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        self.alpha = alpha
        self.fit_prior = fit_prior
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        """Fit Multinomial Naive Bayes classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors (non-negative).
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : MultinomialNB
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        if np.any(X < 0):
            raise ValueError("Negative values in data passed to MultinomialNB")
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Count samples per class
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        
        # Calculate class priors
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_counts / len(y))
        else:
            self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))
        
        # Calculate feature counts per class
        feature_count = np.zeros((self.n_classes_, self.n_features_in_))
        
        for idx, c in enumerate(self.classes_):
            feature_count[idx, :] = X[y == c].sum(axis=0)
        
        # Apply Laplace smoothing and calculate log probabilities
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log-probability of samples for each class.
        """
        X = check_array(X)
        
        if not hasattr(self, 'feature_log_prob_'):
            raise NotFittedError("This MultinomialNB instance is not fitted yet.")
        
        # Log likelihood = sum over features of (count * log_prob)
        log_likelihood = X @ self.feature_log_prob_.T
        
        # Log posterior = log prior + log likelihood
        log_posterior = self.class_log_prior_ + log_likelihood
        
        # Normalize
        log_prob_x = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        
        return log_posterior - log_prob_x
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of samples for each class.
        """
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on an array of test vectors X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class BernoulliNB(BaseClassifier):
    """Bernoulli Naive Bayes classifier.
    
    Suitable for binary/boolean features. Implements the naive Bayes
    algorithm for data that is distributed according to multivariate
    Bernoulli distributions.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
    ):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        """Fit Bernoulli Naive Bayes classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : BernoulliNB
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        # Binarize if needed
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Count samples per class
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        
        # Calculate class priors
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_counts / len(y))
        else:
            self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))
        
        # Count feature occurrences per class
        feature_count = np.zeros((self.n_classes_, self.n_features_in_))
        
        for idx, c in enumerate(self.classes_):
            feature_count[idx, :] = X[y == c].sum(axis=0)
        
        # Apply smoothing
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = class_counts.reshape(-1, 1) + 2 * self.alpha
        
        # Calculate log probabilities
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        self.feature_log_prob_neg_ = np.log(1 - smoothed_fc / smoothed_cc)
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log-probability of samples for each class.
        """
        X = check_array(X)
        
        if not hasattr(self, 'feature_log_prob_'):
            raise NotFittedError("This BernoulliNB instance is not fitted yet.")
        
        # Binarize if needed
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        # For Bernoulli: P(x|y) = product of P(x_i|y)^x_i * P(not x_i|y)^(1-x_i)
        # In log space: sum of x_i * log P(x_i|y) + (1-x_i) * log P(not x_i|y)
        log_likelihood = (
            X @ self.feature_log_prob_.T +
            (1 - X) @ self.feature_log_prob_neg_.T
        )
        
        # Log posterior = log prior + log likelihood
        log_posterior = self.class_log_prior_ + log_likelihood
        
        # Normalize
        log_prob_x = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        
        return log_posterior - log_prob_x
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test vector X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of samples for each class.
        """
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on an array of test vectors X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
