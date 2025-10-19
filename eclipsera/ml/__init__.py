"""Machine learning algorithms module for Eclipsera.

This module contains implementations of classical machine learning algorithms
including linear models, tree-based methods, ensemble methods, SVMs, and more.
"""
from .ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .linear import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from .naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from .neighbors import KNeighborsClassifier, KNeighborsRegressor
from .neural_network import MLPClassifier, MLPRegressor
from .svm import SVC, SVR, LinearSVC, LinearSVR
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    # Linear models
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    # Tree models
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # Ensemble models
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    # SVM models
    "SVC",
    "SVR",
    "LinearSVC",
    "LinearSVR",
    # Naive Bayes
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    # Neighbors
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    # Neural Networks
    "MLPClassifier",
    "MLPRegressor",
]
