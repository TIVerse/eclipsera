<div align="center">

# 🌒 Eclipsera

### *Modern Machine Learning, Built from Scratch*

[![PyPI version](https://img.shields.io/pypi/v/eclipsera?style=flat-square&logo=pypi&logoColor=white&label=PyPI&color=6366f1)](https://pypi.org/project/eclipsera/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/eclipsera?style=flat-square&color=f97316&label=Downloads&logo=pypi)](https://pypi.org/project/eclipsera/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-eab308?style=flat-square&logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/github/actions/workflow/status/tiverse/eclipsera/ci.yml?style=flat-square&logo=github&label=Build&color=6366f1)](https://github.com/tiverse/eclipsera/actions)
[![Coverage](https://img.shields.io/badge/Coverage-88%25-10b981?style=flat-square&logo=codecov&logoColor=white)](https://github.com/tiverse/eclipsera/actions)

**Version 1.2.0** • **68+ Algorithms** • **100% Python** • **Scikit-learn Compatible**

[📖 Documentation](https://tiverse.github.io/eclipsera/) • [🚀 Quick Start](#-quick-start) • [📦 Installation](#-installation) • [🤝 Contributing](#-contributing)

---

</div>

## 🎯 About Eclipsera

**Eclipsera** is a comprehensive machine learning framework built from the ground up in Python. Designed for researchers, developers, and data scientists who need a unified, powerful, and transparent ML ecosystem without the complexity of heavy dependencies.

### ✨ Key Highlights

- **🧠 68+ Algorithms** - From classical ML to modern AutoML
- **🔧 Drop-in Compatible** - 100% Scikit-learn API compatibility
- **⚡ Performance Optimized** - FastAlloc object pooling for 5-15% speedup
- **🛡️ Production Ready** - 88% test coverage with 618 passing tests
- **🔍 Explainable AI** - Built-in model interpretation tools
- **🎨 Modern Python** - Full type hints, Python 3.11+ support

---

## 🚀 Quick Start

### Installation

```bash
# Core package
pip install eclipsera

# With performance optimizations (recommended)
pip install eclipsera[perf]

# For plotting and visualization
pip install eclipsera[plot]

# Everything
pip install eclipsera[all]
```

### First Steps

```python
import numpy as np
from eclipsera.ml import RandomForestClassifier
from eclipsera.model_selection import train_test_split

# Generate sample data
X = np.random.randn(150, 4)
y = np.random.randint(0, 3, 150)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"✅ Accuracy: {score:.3f}")
```

---

## 🌟 Advanced Features

### 🤖 AutoML - Automatic Algorithm Selection

```python
from eclipsera.automl import AutoClassifier

# Let Eclipsera find the best algorithm
auto_clf = AutoClassifier(cv=5, verbose=1)
auto_clf.fit(X_train, y_train)

print(f"🏆 Best: {auto_clf.best_algorithm_} (score: {auto_clf.best_score_:.4f})")
```

### 🔍 Model Explainability

```python
from eclipsera.explainability import permutation_importance

# Understand what drives your model
result = permutation_importance(model, X_test, y_test, n_repeats=10)
for i, importance in enumerate(result['importances_mean']):
    print(f"📊 Feature {i}: {importance:.4f}")
```

### 🔗 Complete ML Pipeline

```python
from eclipsera.pipeline import Pipeline
from eclipsera.preprocessing import StandardScaler
from eclipsera.feature_selection import SelectKBest
from eclipsera.decomposition import PCA
from eclipsera.ml import LogisticRegression

# Build sophisticated workflows
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=20)),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(f"🎯 Pipeline Score: {pipe.score(X_test, y_test):.4f}")
```

---

## 📚 Algorithm Library

### 🎓 Supervised Learning (28 Algorithms)
- **Linear Models:** LogisticRegression, LinearRegression, Ridge, Lasso
- **Tree-Based:** DecisionTree, RandomForest, GradientBoosting
- **Support Vector Machines:** SVC, SVR (linear, rbf, poly, sigmoid)
- **Naive Bayes:** GaussianNB, MultinomialNB, BernoulliNB
- **Nearest Neighbors:** KNeighborsClassifier, KNeighborsRegressor
- **Neural Networks:** MLPClassifier, MLPRegressor

### 🎯 Clustering (7 Methods)
- **K-Means** (Standard & MiniBatch)
- **DBSCAN** (Density-based)
- **Agglomerative** (4 linkage methods)
- **Spectral** (RBF & k-NN affinity)
- **MeanShift** (Kernel density)
- **Gaussian Mixture** (Probabilistic EM)

### 📉 Dimensionality Reduction
- **PCA**, **TruncatedSVD**, **NMF**

### 🗺️ Manifold Learning
- **t-SNE**, **Isomap**, **LocallyLinearEmbedding**

### ⚙️ Preprocessing & Feature Selection
- **Scalers:** StandardScaler, MinMaxScaler, RobustScaler
- **Imputers:** SimpleImputer, KNNImputer
- **Encoders:** LabelEncoder, OneHotEncoder, OrdinalEncoder
- **Selection:** VarianceThreshold, SelectKBest, RFE

### 🤖 AutoML & Explainability
- **AutoClassifier**, **AutoRegressor**
- **permutation_importance**, **partial_dependence**
- **plot_partial_dependence**, **get_feature_importance**

---

## 📊 Project Metrics

| Metric | Value |
|:-------|:------|
| 🎯 **Total Algorithms** | 68+ |
| 📝 **Lines of Code** | ~10,500 |
| ✅ **Test Coverage** | 88% |
| 🧪 **Tests Passing** | 618/618 |
| 📦 **Python Version** | 3.11+ |
| 🔗 **Dependencies** | NumPy, SciPy, Pandas, Joblib |

---

## 🔄 Version History

### 🛡️ Version 1.2.0 - Security Hardening (Current)
**Security & Code Quality Focus**
- ✅ Security hardening with pickle deserialization controls
- ✅ GitHub Actions supply chain hardening (pinned SHAs)
- ✅ Added Bandit and detect-secrets security scanning
- ✅ Fixed 522 flake8 linting errors
- ✅ SBOM generation for supply chain transparency
- ✅ Enhanced CLI security with confirmation gates

### 📈 Previous Versions
- **v1.1.0** - Performance optimizations with FastAlloc
- **v1.0.0** - Initial stable release

[📋 Full Changelog](https://github.com/tiverse/eclipsera/blob/master/versions/v1.2.0.md)

---

## 🛠️ Development

### Setup for Contributors

```bash
git clone https://github.com/tiverse/eclipsera.git
cd eclipsera
pip install -e ".[dev,perf]"
pytest tests/
```

### Contributing Guidelines

We welcome all contributions! Here's how you can help:

- 🐛 **Report bugs** - Found an issue? Open an issue
- ✨ **Request features** - Have an idea? Share it
- 📝 **Improve docs** - Help others understand Eclipsera
- 🧪 **Add tests** - Improve our coverage
- 💻 **Write code** - Add algorithms or optimize existing ones

---

## 📄 License & Citation

### License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Citation
If you use Eclipsera in your research, please cite:

```bibtex
@software{eclipsera2024,
  title = {Eclipsera: A Modern Machine Learning Framework},
  author = {Roy, Eshan},
  year = {2024},
  url = {https://github.com/tiverse/eclipsera},
  version = {1.2.0}
}
```

---

## 🔗 Useful Links

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-8b5cf6?style=flat-square&logo=readthedocs&logoColor=white)](https://tiverse.github.io/eclipsera/)
[![Performance Guide](https://img.shields.io/badge/Performance_Guide-10b981?style=flat-square&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/blob/master/FASTALLOC_USAGE.md)
[![Roadmap](https://img.shields.io/badge/Roadmap-f97316?style=flat-square&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/blob/master/milestone/ROADMAP_2.0.0.md)
[![Issues](https://img.shields.io/badge/Issues-ec4899?style=flat-square&logo=github&logoColor=white)](https://github.com/tiverse/eclipsera/issues)

</div>

---

<div align="center">

### 🌟 Why Eclipsera?

| 🎯 **Comprehensive** | 🔧 **Compatible** | 🛡️ **Reliable** |
|:---------------------|:------------------|:-----------------|
| 68 algorithms covering all major ML workflows | 100% Scikit-learn compatible API | 88% test coverage with 618 passing tests |

| 🎨 **Modern** | ⚡ **Lightweight** | 🔍 **Transparent** |
|:--------------|:-------------------|:-------------------|
| Built for Python 3.11+ with complete type hints | Minimal dependencies - only NumPy, SciPy, Pandas, Joblib | Clear, documented implementations |

---

**Built with ❤️ by [Eshan Roy](https://github.com/tiverse)**

*Empowering the next generation of machine learning applications*

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-ec4899?style=flat-square)
![Python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white)
![Open Source](https://img.shields.io/badge/Open%20Source-14b8a6?style=flat-square&logo=open-source-initiative&logoColor=white)

⭐ **If you find Eclipsera useful, consider giving it a star!** ⭐

</div>