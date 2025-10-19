# Contributing to Eclipsera

Thank you for your interest in contributing to Eclipsera! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/eclipsera.git
   cd eclipsera
   ```

2. **Create Virtual Environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code following PEP 8
- Add type hints to all public APIs
- Write Google-style docstrings
- Ensure backward compatibility where possible

### 3. Add Tests

- Write unit tests for new functionality
- Aim for ≥90% code coverage
- Include edge cases and error conditions
- Run tests locally:
  ```bash
  pytest tests/
  pytest --cov=eclipsera --cov-report=html
  ```

### 4. Update Documentation

- Update docstrings
- Add examples to relevant documentation
- Update CHANGELOG.md
- Build docs locally:
  ```bash
  cd docs
  make html
  ```

### 5. Run Quality Checks

```bash
# Format code
black eclipsera tests
isort eclipsera tests

# Lint
flake8 eclipsera tests

# Type check
mypy eclipsera

# Or let pre-commit run all checks
pre-commit run --all-files
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new feature X"  # Use conventional commits
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 7. Push and Create PR

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub.

## 🎯 Coding Standards

### Code Style

- **PEP 8** compliance (enforced by black and flake8)
- **Line length**: 100 characters
- **Imports**: Organized with isort
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style, required for all public classes/methods/functions

### Docstring Example

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = None
) -> RandomForestClassifier:
    """Train a random forest classifier.

    This function trains a random forest classifier on the provided
    training data with specified hyperparameters.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target labels.
    n_estimators : int, default=100
        Number of trees in the forest.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Trained model instance.

    Raises
    ------
    ValueError
        If X and y have incompatible shapes.

    Examples
    --------
    >>> from eclipsera.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = train_model(X, y, n_estimators=50)
    >>> model.score(X, y)
    0.98
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model
```

### Testing Guidelines

- **Coverage**: Minimum 90% for new code
- **Test naming**: `test_<function_name>_<scenario>`
- **Fixtures**: Use pytest fixtures for common setup
- **Markers**: Use `@pytest.mark.slow` for slow tests, `@pytest.mark.gpu` for GPU tests

Example test:
```python
import pytest
import numpy as np
from eclipsera.ml.linear import LinearRegression


def test_linear_regression_fit_predict():
    """Test LinearRegression fit and predict methods."""
    X = np.random.randn(100, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert predictions.shape == (100,)
    assert model.score(X, y) > 0.95


def test_linear_regression_invalid_input():
    """Test LinearRegression raises error on invalid input."""
    model = LinearRegression()
    
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]), np.array([1, 2]))
```

## 🐛 Bug Reports

### Before Submitting

1. Check existing issues
2. Verify it's reproducible on the latest version
3. Try to isolate the bug with minimal code

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
```python
# Minimal code to reproduce
from eclipsera import ...
...
```

**Expected behavior**
What you expected to happen.

**Environment:**
- Eclipsera version: [e.g., 0.1.0]
- Python version: [e.g., 3.12.0]
- OS: [e.g., Ubuntu 22.04]
- Dependencies: [relevant package versions]

**Additional context**
Any other relevant information.
```

## ✨ Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is planned
2. Describe the use case and motivation
3. Provide example API usage
4. Consider implementation complexity

## 📝 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines (black, isort, flake8 pass)
- [ ] Type hints added for public APIs
- [ ] Google-style docstrings added
- [ ] Tests added with ≥90% coverage
- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Pre-commit hooks pass
- [ ] Commits follow conventional commit format
- [ ] PR description clearly explains changes

## 🔍 Code Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Review**: At least one maintainer review required
3. **Changes**: Address review feedback
4. **Merge**: Maintainer will merge when approved

## 🎓 Resources

- **Python Style Guide**: [PEP 8](https://pep8.org/)
- **Type Hints**: [PEP 484](https://www.python.org/dev/peps/pep-0484/)
- **Docstrings**: [PEP 257](https://www.python.org/dev/peps/pep-0257/)
- **Google Docstring Style**: [Example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/tiverse/eclipsera/issues)
- **Email**: eshanized@proton.me

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity.

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community

**Unacceptable behavior:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct reasonably considered inappropriate

### Enforcement

Instances of abusive behavior may be reported to eshanized@proton.me. All complaints will be reviewed and investigated.

---

Thank you for contributing to Eclipsera! 🌒
