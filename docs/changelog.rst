Changelog
=========

All notable changes to Eclipsera are documented here.

Version 1.1.0 (2024-10-20)
--------------------------

**New Features:**

* Added ``AutoClassifier`` for automatic classification algorithm selection
* Added ``AutoRegressor`` for automatic regression algorithm selection  
* Added ``permutation_importance`` for model-agnostic feature importance
* Added ``partial_dependence`` for feature effect analysis
* Added ``plot_partial_dependence`` for PD visualization
* Added ``get_feature_importance`` for universal importance extraction

**Improvements:**

* Enhanced documentation with comprehensive tutorials
* Improved API consistency across modules
* Added 42 new tests for AutoML and Explainability

**Statistics:**

* Total algorithms: 68 (up from 62)
* Test coverage: 88%
* Total tests: 618 (up from 576)

Version 1.0.0 (2024-10-19)
--------------------------

**New Features:**

* Added Isomap manifold learning
* Added LocallyLinearEmbedding (LLE)
* Added GaussianMixture probabilistic clustering

**Modules:**

* 62 total algorithms across 10 modules
* 576 tests with 88% coverage

Version 0.6.0 (2024-10-18)
--------------------------

**New Features:**

* Added t-SNE for manifold learning
* Added Spectral Clustering
* Added Mean Shift clustering

Version 0.5.0 (2024-10-17)
--------------------------

**New Features:**

* Added Hierarchical/Agglomerative clustering
* Added NMF (Non-negative Matrix Factorization)
* Added RFE (Recursive Feature Elimination)

Version 0.4.0 (2024-10-16)
--------------------------

**New Features:**

* Added K-Means and MiniBatch K-Means clustering
* Added DBSCAN clustering
* Added PCA and TruncatedSVD
* Added basic feature selection tools

Version 0.3.0 (2024-10-15)
--------------------------

**New Features:**

* Added cross-validation utilities
* Added GridSearchCV and RandomizedSearchCV
* Added Pipeline and FeatureUnion

Version 0.2.0 (2024-10-14)
--------------------------

**New Features:**

* Added 28 supervised learning algorithms:
  * Linear models (4)
  * Tree-based models (6)
  * SVMs (2)
  * Naive Bayes (3)
  * K-Nearest Neighbors (2)
  * Neural Networks (2)

Version 0.1.0 (2024-10-13)
--------------------------

**Initial Release:**

* Core infrastructure
* Base classes and utilities
* Metrics and validation
* Preprocessing tools (10)
