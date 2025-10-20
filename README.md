# 🌒 Eclipsera

**A Modern Machine Learning Framework for Python**

[![PyPI version](https://img.shields.io/pypi/v/eclipsera?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI%20Version)](https://pypi.org/project/eclipsera/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eclipsera?style=for-the-badge&color=orange&label=Downloads&logo=pypi)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-88%25-brightgreen?style=for-the-badge&logo=codecov&logoColor=white)
[![Build](https://img.shields.io/github/actions/workflow/status/tiverse/eclipsera/ci.yml?style=for-the-badge&logo=github&label=Build)](https://github.com/tiverse/eclipsera/actions)
[![Docs](https://img.shields.io/badge/Docs-Online-blueviolet?style=for-the-badge&logo=readthedocs&logoColor=white)](https://tiverse.github.io/eclipsera/)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/tiverse/eclipsera?style=for-the-badge&logo=codefactor&logoColor=white)](https://www.codefactor.io/repository/github/tiverse/eclipsera)
![Typing](https://img.shields.io/badge/Type%20Checked-mypy-informational?style=for-the-badge&logo=python&logoColor=white)
![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

---

## 🚀 Overview

**Eclipsera** is a comprehensive machine learning framework built from scratch with 68 algorithms spanning classical ML, clustering, dimensionality reduction, manifold learning, AutoML, and explainability.

**Features:**
- 🤖 **AutoML** — Automatic algorithm selection and optimization
- 🔍 **Explainability** — Permutation importance, partial dependence, feature importance
- 📊 **Supervised Learning** — 28 classification and regression algorithms
- 🎯 **Clustering** — 7 algorithms including K-Means, DBSCAN, Spectral, Gaussian Mixture
- 📉 **Dimensionality Reduction** — PCA, NMF, TruncatedSVD
- 🗺️ **Manifold Learning** — t-SNE, Isomap, LLE
- ⚙️ **Feature Selection** — Variance threshold, univariate selection, RFE
- 🔧 **Preprocessing** — Scalers, imputers, encoders
- 🔬 **Model Selection** — Cross-validation, hyperparameter search
- 🔗 **Pipelines** — Composable ML workflows

---

## ⚡ Quick Start

```python
import numpy as np
from eclipsera.ml import RandomForestClassifier
from eclipsera.model_selection import train_test_split

# Load data
X = np.random.randn(150, 4)
y = np.random.randint(0, 3, 150)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

### AutoML Example

```python
from eclipsera.automl import AutoClassifier

# Automatically select best algorithm
auto_clf = AutoClassifier(cv=5, verbose=1)
auto_clf.fit(X_train, y_train)

print(f"Best algorithm: {auto_clf.best_algorithm_}")
print(f"Best score: {auto_clf.best_score_:.4f}")

# Use like any classifier
y_pred = auto_clf.predict(X_test)
```

### Explainability Example

```python
from eclipsera.explainability import permutation_importance
from eclipsera.ml import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)

# Compute feature importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10)

for i in range(X.shape[1]):
    print(f"Feature {i}: {result['importances_mean'][i]:.4f}")
```

---

## 📦 Installation

```bash
pip install eclipsera
```

**Requirements:**
- Python 3.8+
- NumPy
- SciPy
- matplotlib (optional, for plotting)

**From source:**
```bash
git clone https://github.com/tiverse/eclipsera.git
cd eclipsera
pip install -e .
```

---

## 🎯 Complete Feature List

### Supervised Learning (28 algorithms)

**Linear Models:**
- LogisticRegression, LinearRegression
- Ridge, Lasso

**Tree-Based Models:**
- DecisionTreeClassifier, DecisionTreeRegressor
- RandomForestClassifier, RandomForestRegressor
- GradientBoostingClassifier, GradientBoostingRegressor

**Support Vector Machines:**
- SVC (kernels: linear, rbf, poly, sigmoid)
- SVR

**Naive Bayes:**
- GaussianNB, MultinomialNB, BernoulliNB

**Nearest Neighbors:**
- KNeighborsClassifier, KNeighborsRegressor

**Neural Networks:**
- MLPClassifier, MLPRegressor

### Clustering (7 algorithms)
- KMeans, MiniBatchKMeans
- DBSCAN
- AgglomerativeClustering (4 linkage methods)
- SpectralClustering (RBF & k-NN affinity)
- MeanShift
- GaussianMixture (probabilistic)

### Dimensionality Reduction (3 algorithms)
- PCA (Principal Component Analysis)
- TruncatedSVD
- NMF (Non-negative Matrix Factorization)

### Manifold Learning (3 algorithms)
- TSNE (t-distributed Stochastic Neighbor Embedding)
- Isomap (Isometric Mapping)
- LocallyLinearEmbedding (LLE)

### Feature Selection (3 tools)
- VarianceThreshold
- SelectKBest (with f_classif, chi2)
- RFE (Recursive Feature Elimination)

### Preprocessing (10 tools)
- StandardScaler, MinMaxScaler, RobustScaler
- SimpleImputer (4 strategies)
- KNNImputer
- LabelEncoder, OneHotEncoder, OrdinalEncoder

### AutoML (2 tools)
- AutoClassifier — Automatic classification
- AutoRegressor — Automatic regression

### Explainability (4 tools)
- permutation_importance — Feature importance
- partial_dependence — Feature effect analysis
- plot_partial_dependence — Visualization
- get_feature_importance — Universal extraction

### Model Selection (8 utilities)
- train_test_split
- KFold, StratifiedKFold
- cross_val_score, cross_validate
- GridSearchCV, RandomizedSearchCV

### Pipeline (3 tools)
- Pipeline
- FeatureUnion
- make_pipeline

---

## 📚 Usage Examples

### Complete Pipeline

```python
from eclipsera.pipeline import Pipeline
from eclipsera.preprocessing import StandardScaler
from eclipsera.feature_selection import SelectKBest
from eclipsera.decomposition import PCA
from eclipsera.ml import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=20)),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression())
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

### Clustering

```python
from eclipsera.cluster import GaussianMixture

# Probabilistic clustering
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)
probabilities = gmm.predict_proba(X)
```

### Manifold Learning

```python
from eclipsera.manifold import TSNE

# Non-linear dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(X_high_dim)
```

### Hyperparameter Optimization

```python
from eclipsera.model_selection import GridSearchCV
from eclipsera.ml import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

---

## 📊 Project Statistics

- **Total Features**: 68 algorithms/tools
- **Lines of Code**: ~10,500
- **Test Coverage**: 88%
- **Total Tests**: 618 (passing)
- **Modules**: 12
- **Python Version**: 3.8+

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Additional algorithms
- Performance optimizations
- Documentation improvements
- Bug fixes
- Example notebooks

### Development Setup
```bash
git clone https://github.com/tiverse/eclipsera.git
cd eclipsera
pip install -e .
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Citation

If you use Eclipsera in your research, please cite:

```bibtex
@software{eclipsera2024,
  title = {Eclipsera: A Modern Machine Learning Framework},
  author = {Roy, Eshan},
  year = {2024},
  url = {https://github.com/tiverse/eclipsera},
  version = {1.1.0}
}
```

---

## 🔗 Links

- **Homepage**: [https://github.com/tiverse/eclipsera](https://github.com/tiverse/eclipsera)
- **Issues**: [https://github.com/tiverse/eclipsera/issues](https://github.com/tiverse/eclipsera/issues)

## 🌟 Why Eclipsera?

- **Comprehensive**: 68 algorithms covering all major ML workflows
- **100% Scikit-learn Compatible**: Drop-in replacement for most use cases
- **Type-Safe**: Complete type hints throughout
- **Well-Documented**: Google-style docstrings for all APIs
- **Tested**: 88% test coverage with 618 passing tests
- **Modern**: Built for Python 3.8+ with best practices
- **Minimal Dependencies**: Only NumPy and SciPy required
- **Extensible**: Easy to add custom estimators

---

**Built with precision by Eshan Roy**
