<div align="center">

# 🌒 Eclipsera

### *Where Machine Learning Meets Innovation*

<!-- <img src="https://raw.githubusercontent.com/tiverse/eclipsera/main/assets/logo.png" alt="Eclipsera Logo" width="200"/> -->

**A next-generation Machine Learning framework built from scratch**  
*68+ algorithms | 100% Python | Scikit-learn Compatible*

<br>

[![PyPI version](https://img.shields.io/pypi/v/eclipsera?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI%20Version&color=6366f1)](https://pypi.org/project/eclipsera/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eclipsera?style=for-the-badge&color=f97316&label=Downloads&logo=pypi)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-eab308?style=for-the-badge&logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)

![Tests](https://img.shields.io/badge/Tests-618_Passing-10b981?style=for-the-badge&logo=pytest&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-88%25-10b981?style=for-the-badge&logo=codecov&logoColor=white)
[![Build](https://img.shields.io/github/actions/workflow/status/tiverse/eclipsera/ci.yml?style=for-the-badge&logo=github&label=Build&color=6366f1)](https://github.com/tiverse/eclipsera/actions)
[![Docs](https://img.shields.io/badge/Docs-Online-8b5cf6?style=for-the-badge&logo=readthedocs&logoColor=white)](https://tiverse.github.io/eclipsera/)

[![CodeFactor](https://img.shields.io/codefactor/grade/github/tiverse/eclipsera?style=for-the-badge&logo=codefactor&logoColor=white&color=14b8a6)](https://www.codefactor.io/repository/github/tiverse/eclipsera)
![Typing](https://img.shields.io/badge/Type%20Checked-mypy-0ea5e9?style=for-the-badge&logo=python&logoColor=white)
![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-ec4899?style=for-the-badge)

<br>

[🚀 Quick Start](#-quick-start) • [📦 Installation](#-installation) • [✨ Features](#-key-features) • [📖 Docs](https://tiverse.github.io/eclipsera/) • [🤝 Contributing](#-contributing)

---

</div>

## 🎯 What is Eclipsera?

> **Eclipsera** empowers researchers, developers, and data scientists with a **unified ML ecosystem** built entirely from scratch. Train, evaluate, and explain models with elegant, intuitive APIs — no compromises on power or flexibility.

<br>

<table>
<tr>
<td width="50%">

### 🔥 Why Choose Eclipsera?

✨ **68+ Algorithms** - Classical ML to modern AutoML  
🧩 **Modular Design** - Mix and match components effortlessly  
🎯 **100% Compatible** - Drop-in Scikit-learn replacement  
🔍 **Built-in Explainability** - Understand your models deeply  
⚡ **Pure Python** - No heavy dependencies  
🚀 **Performance Optimized** - Object pooling with FastAlloc for 5-15% speedup  
🛡️ **Type-Safe** - Complete type hints throughout  
📊 **Production-Ready** - 88% test coverage, battle-tested  

</td>
<td width="50%">

### 💡 Perfect For

🔬 **Researchers** - Experiment with cutting-edge algorithms  
👨‍💻 **Developers** - Build ML pipelines with confidence  
📈 **Data Scientists** - Prototype and deploy faster  
🎓 **Students** - Learn ML from transparent implementations  
🏢 **Teams** - Standardize your ML workflow  

</td>
</tr>
</table>

---

## ✨ Key Features

<div align="center">

| 🧠 **Capability** | 🚀 **What You Get** |
|:------------------|:--------------------|
| **🤖 AutoML** | Automatic algorithm selection, hyperparameter tuning, and optimization |
| **🔍 Explainability** | Permutation importance, partial dependence, feature importance analysis |
| **📊 Supervised Learning** | 28 algorithms: Linear models, Trees, SVM, Neural Networks, and more |
| **🎯 Clustering** | 7 methods: K-Means, DBSCAN, Spectral, Gaussian Mixture, MeanShift |
| **📉 Dimensionality Reduction** | PCA, NMF, TruncatedSVD for efficient feature compression |
| **🗺️ Manifold Learning** | t-SNE, Isomap, LLE for non-linear embeddings |
| **⚙️ Feature Selection** | Variance thresholding, univariate selection, RFE |
| **🔧 Preprocessing** | Scalers, imputers, encoders for robust data preparation |
| **🔬 Model Selection** | Cross-validation, grid/random search with performance optimizations |
| **🚀 Performance** | Object pooling with FastAlloc for 5-15% speedup in CV/search operations |
| **🔗 Pipelines** | Composable workflows with reusable, modular components |

</div>

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation
pip install eclipsera

# With performance optimizations (recommended)
pip install eclipsera[perf]
```

<details>
<summary>📦 <b>From Source</b></summary>

```bash
git clone https://github.com/tiverse/eclipsera.git
cd eclipsera
pip install -e ".[dev]"  # Development version
pip install -e ".[perf]"  # With performance optimizations
```
</details>

<br>

### 🎯 Basic Usage

```python
import numpy as np
from eclipsera.ml import RandomForestClassifier
from eclipsera.model_selection import train_test_split

# Prepare data
X = np.random.randn(150, 4)
y = np.random.randint(0, 3, 150)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"🎯 Accuracy: {score:.3f}")
```

---

## 🌟 Advanced Examples

<details open>
<summary><b>🤖 AutoML - Automatic Algorithm Selection</b></summary>

```python
from eclipsera.automl import AutoClassifier

# Let Eclipsera choose the best algorithm for you
auto_clf = AutoClassifier(cv=5, verbose=1)
auto_clf.fit(X_train, y_train)

print(f"✅ Best algorithm: {auto_clf.best_algorithm_}")
print(f"📊 Best score: {auto_clf.best_score_:.4f}")

# Use it like any other classifier
y_pred = auto_clf.predict(X_test)
```
</details>

<details>
<summary><b>🔍 Explainability - Understand Your Models</b></summary>

```python
from eclipsera.explainability import permutation_importance
from eclipsera.ml import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)

# Compute feature importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10)

for i in range(X.shape[1]):
    print(f"📌 Feature {i}: {result['importances_mean'][i]:.4f}")
```
</details>

<details>
<summary><b>🔗 Complete ML Pipeline</b></summary>

```python
from eclipsera.pipeline import Pipeline
from eclipsera.preprocessing import StandardScaler
from eclipsera.feature_selection import SelectKBest
from eclipsera.decomposition import PCA
from eclipsera.ml import LogisticRegression

# Build a sophisticated pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=20)),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression())
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(f"🎯 Pipeline Score: {score:.4f}")
```
</details>

<details>
<summary><b>🎯 Clustering with Gaussian Mixture</b></summary>

```python
from eclipsera.cluster import GaussianMixture

# Probabilistic clustering
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)
probabilities = gmm.predict_proba(X)

print(f"🏷️ Cluster assignments: {labels}")
print(f"📊 Confidence scores: {probabilities}")
```
</details>

<details>
<summary><b>🗺️ Manifold Learning with t-SNE</b></summary>

```python
from eclipsera.manifold import TSNE

# Non-linear dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(X_high_dim)

# Visualize high-dimensional data in 2D
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
plt.title("t-SNE Visualization")
plt.show()
```
</details>

<details>
<summary><b>⚙️ Hyperparameter Optimization</b></summary>

```python
from eclipsera.model_selection import GridSearchCV
from eclipsera.ml import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"✨ Best params: {grid.best_params_}")
print(f"🏆 Best score: {grid.best_score_:.4f}")
```
</details>

<details>
<summary>🚀 <b>Performance Optimizations with FastAlloc</b></summary>

```python
# Install with performance extras
pip install eclipsera[perf]

# Automatic object pooling in cross-validation and hyperparameter search
from eclipsera.model_selection import GridSearchCV, cross_val_score
from eclipsera.ml import LogisticRegression

# These operations are now 5-15% faster with object pooling!
param_grid = {'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X, y)  # Automatically uses FastAlloc pooling

scores = cross_val_score(LogisticRegression(), X, y, cv=5)  # Also pooled!

# Check if FastAlloc is active
try:
    import fastalloc
    print("✅ FastAlloc is active - performance optimizations enabled!")
except ImportError:
    print("ℹ️ FastAlloc not installed - install with pip install eclipsera[perf]")
```
</details>

---

## 📚 Complete Algorithm Library

<div align="center">

### 🎓 Supervised Learning (28 Algorithms)

</div>

| **Category** | **Algorithms** |
|:-------------|:---------------|
| 📈 **Linear Models** | LogisticRegression • LinearRegression • Ridge • Lasso |
| 🌳 **Tree-Based** | DecisionTree (Clf/Reg) • RandomForest (Clf/Reg) • GradientBoosting (Clf/Reg) |
| 🎯 **Support Vector Machines** | SVC (linear, rbf, poly, sigmoid) • SVR |
| 📊 **Naive Bayes** | GaussianNB • MultinomialNB • BernoulliNB |
| 🎲 **Nearest Neighbors** | KNeighborsClassifier • KNeighborsRegressor |
| 🧠 **Neural Networks** | MLPClassifier • MLPRegressor |

<div align="center">

### 🎯 Clustering (7 Algorithms)

</div>

| **Method** | **Description** |
|:-----------|:----------------|
| 📍 **K-Means** | Standard & MiniBatch variants |
| 🔍 **DBSCAN** | Density-based clustering |
| 🌲 **Agglomerative** | 4 linkage methods (ward, complete, average, single) |
| 🌈 **Spectral** | RBF & k-NN affinity matrices |
| 📊 **MeanShift** | Kernel density estimation |
| 🎲 **Gaussian Mixture** | Probabilistic clustering with EM |

<div align="center">

### 📉 Dimensionality Reduction (3 Tools)

</div>

**PCA** • **TruncatedSVD** • **NMF**

<div align="center">

### 🗺️ Manifold Learning (3 Tools)

</div>

**t-SNE** • **Isomap** • **LocallyLinearEmbedding (LLE)**

<div align="center">

### ⚙️ Feature Selection (3 Tools)

</div>

**VarianceThreshold** • **SelectKBest** (f_classif, chi2) • **RFE**

<div align="center">

### 🔧 Preprocessing (10 Tools)

</div>

**Scalers:** StandardScaler • MinMaxScaler • RobustScaler  
**Imputers:** SimpleImputer (4 strategies) • KNNImputer  
**Encoders:** LabelEncoder • OneHotEncoder • OrdinalEncoder

<div align="center">

### 🤖 AutoML (2 Tools)

</div>

**AutoClassifier** • **AutoRegressor**

<div align="center">

### 🔍 Explainability (4 Tools)

</div>

**permutation_importance** • **partial_dependence** • **plot_partial_dependence** • **get_feature_importance**

<div align="center">

### 🔬 Model Selection (8 Utilities)

</div>

**train_test_split** • **KFold** • **StratifiedKFold** • **cross_val_score** • **cross_validate** • **GridSearchCV** • **RandomizedSearchCV**

<div align="center">

### 🔗 Pipeline (3 Tools)

</div>

**Pipeline** • **FeatureUnion** • **make_pipeline**

---

## 📊 Project at a Glance

<div align="center">

| Metric | Value |
|:-------|:------|
| 🎯 **Total Algorithms** | 68+ |
| 📝 **Lines of Code** | ~10,500 |
| ✅ **Test Coverage** | 88% |
| 🧪 **Total Tests** | 618 (all passing) |
| 📦 **Modules** | 12 |
| 🐍 **Python Version** | 3.11+ |
| 📚 **Dependencies** | NumPy, SciPy (+ optional matplotlib, fastalloc for performance) |

</div>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 **Bug Reports** - Found an issue? Let us know!
- ✨ **Feature Requests** - Have an idea? Share it!
- 📝 **Documentation** - Help others understand Eclipsera
- 🧪 **Tests** - Improve our coverage
- 💻 **Code** - Add new algorithms or optimize existing ones

### Development Setup

```bash
git clone https://github.com/tiverse/eclipsera.git
cd eclipsera
pip install -e ".[dev,perf]"  # Full development setup with performance
pytest tests/
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

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

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera)
[![Documentation](https://img.shields.io/badge/Documentation-8b5cf6?style=for-the-badge&logo=readthedocs&logoColor=white)](https://tiverse.github.io/eclipsera/)
[![Performance Guide](https://img.shields.io/badge/Performance_Guide-10b981?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/blob/main/FASTALLOC_USAGE.md)
[![Roadmap](https://img.shields.io/badge/Roadmap-f97316?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/blob/main/milestone/ROADMAP_2.0.0.md)
[![Issues](https://img.shields.io/badge/Issues-ec4899?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/issues)
[![PyPI](https://img.shields.io/badge/PyPI-f97316?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/eclipsera/)

</div>

---

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tiverse/eclipsera&type=Date)](https://star-history.com/#tiverse/eclipsera&Date)

---

## 💫 Why Eclipsera?

<table>
<tr>
<td align="center" width="33%">
  
### 🎯 Comprehensive
68 algorithms covering all major ML workflows - from preprocessing to deployment

</td>
<td align="center" width="33%">
  
### 🔧 Compatible
100% Scikit-learn compatible API - switch seamlessly between libraries

</td>
<td align="center" width="33%">
  
### 🛡️ Reliable
88% test coverage with 618 passing tests ensures production-ready code

</td>
</tr>
<tr>
<td align="center" width="33%">
  
### 🎨 Modern
Built for Python 3.11+ with complete type hints and best practices

</td>
<td align="center" width="33%">
  
### ⚡ Lightweight
Minimal dependencies - only NumPy and SciPy required for core functionality

</td>
<td align="center" width="33%">
  
### 🔍 Transparent
Understand what's happening under the hood with clear, documented implementations

</td>
</tr>
</table>

---

<br>

**Built with precision and passion by [Eshan Roy](https://github.com/tiverse)**

*Empowering the next generation of machine learning applications*

<br>

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-ec4899?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Open Source](https://img.shields.io/badge/Open%20Source-14b8a6?style=for-the-badge&logo=open-source-initiative&logoColor=white)

---

⭐ **If you find Eclipsera useful, consider giving it a star!** ⭐

</div>