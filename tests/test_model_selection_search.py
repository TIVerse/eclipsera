"""Tests for hyperparameter search with cross-validation."""
import numpy as np
import pytest

from eclipsera.model_selection import GridSearchCV, RandomizedSearchCV
from eclipsera.ml import LogisticRegression, RandomForestClassifier, Ridge


def test_grid_search_cv_basic(binary_classification_data):
    """Test basic GridSearchCV functionality."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0, 10.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=3)
    grid.fit(X, y)
    
    assert hasattr(grid, 'best_params_')
    assert hasattr(grid, 'best_score_')
    assert hasattr(grid, 'best_estimator_')
    assert grid.best_params_['C'] in [0.1, 1.0, 10.0]


def test_grid_search_cv_predict(binary_classification_data):
    """Test GridSearchCV predict method."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2)
    grid.fit(X, y)
    
    y_pred = grid.predict(X)
    assert y_pred.shape == y.shape


def test_grid_search_cv_score(binary_classification_data):
    """Test GridSearchCV score method."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2)
    grid.fit(X, y)
    
    score = grid.score(X, y)
    assert 0 <= score <= 1


def test_grid_search_cv_multiple_params(binary_classification_data):
    """Test GridSearchCV with multiple parameters."""
    X, y = binary_classification_data
    
    param_grid = {
        'n_estimators': [5, 10],
        'max_depth': [3, 5]
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2)
    grid.fit(X, y)
    
    assert 'n_estimators' in grid.best_params_
    assert 'max_depth' in grid.best_params_


def test_grid_search_cv_results(binary_classification_data):
    """Test GridSearchCV cv_results_ attribute."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0, 10.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=3)
    grid.fit(X, y)
    
    assert 'params' in grid.cv_results_
    assert 'mean_test_score' in grid.cv_results_
    assert 'std_test_score' in grid.cv_results_
    assert len(grid.cv_results_['params']) == 3


def test_grid_search_cv_no_refit(binary_classification_data):
    """Test GridSearchCV with refit=False."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2, refit=False)
    grid.fit(X, y)
    
    assert hasattr(grid, 'best_params_')
    assert hasattr(grid, 'best_estimator_')


def test_grid_search_cv_verbose(binary_classification_data, capsys):
    """Test GridSearchCV verbose mode."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2, verbose=1)
    grid.fit(X, y)
    
    captured = capsys.readouterr()
    assert 'Fitting' in captured.out


def test_grid_search_cv_regression(regression_data):
    """Test GridSearchCV on regression task."""
    X, y = regression_data
    
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    
    # Use KFold for regression, not StratifiedKFold
    from eclipsera.model_selection import KFold
    grid = GridSearchCV(Ridge(), param_grid, cv=KFold(n_splits=3))
    grid.fit(X, y)
    
    assert hasattr(grid, 'best_params_')
    y_pred = grid.predict(X)
    assert y_pred.shape == y.shape


def test_grid_search_cv_list_of_dicts(binary_classification_data):
    """Test GridSearchCV with list of parameter grids."""
    X, y = binary_classification_data
    
    param_grid = [
        {'C': [0.1, 1.0]},
        {'C': [10.0, 100.0]}
    ]
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2)
    grid.fit(X, y)
    
    assert hasattr(grid, 'best_params_')
    assert len(grid.cv_results_['params']) == 4


def test_randomized_search_cv_basic(binary_classification_data):
    """Test basic RandomizedSearchCV functionality."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=3,
        cv=2,
        random_state=42
    )
    search.fit(X, y)
    
    assert hasattr(search, 'best_params_')
    assert hasattr(search, 'best_score_')
    assert hasattr(search, 'best_estimator_')


def test_randomized_search_cv_n_iter(binary_classification_data):
    """Test RandomizedSearchCV samples n_iter combinations."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0, 10.0, 100.0, 1000.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=3,
        cv=2,
        random_state=42
    )
    search.fit(X, y)
    
    # Should try exactly n_iter combinations
    assert len(search.cv_results_['params']) == 3


def test_randomized_search_cv_predict(binary_classification_data):
    """Test RandomizedSearchCV predict method."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0, 10.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=2,
        cv=2,
        random_state=42
    )
    search.fit(X, y)
    
    y_pred = search.predict(X)
    assert y_pred.shape == y.shape


def test_randomized_search_cv_score(binary_classification_data):
    """Test RandomizedSearchCV score method."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0, 10.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=2,
        cv=2,
        random_state=42
    )
    search.fit(X, y)
    
    score = search.score(X, y)
    assert 0 <= score <= 1


def test_randomized_search_cv_verbose(binary_classification_data, capsys):
    """Test RandomizedSearchCV verbose mode."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=2,
        cv=2,
        verbose=1,
        random_state=42
    )
    search.fit(X, y)
    
    captured = capsys.readouterr()
    assert 'Fitting' in captured.out


def test_randomized_search_cv_regression(regression_data):
    """Test RandomizedSearchCV on regression task."""
    X, y = regression_data
    
    param_dist = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    
    # Use KFold for regression
    from eclipsera.model_selection import KFold
    search = RandomizedSearchCV(Ridge(), param_dist, n_iter=3, cv=KFold(n_splits=2), random_state=42)
    search.fit(X, y)
    
    assert hasattr(search, 'best_params_')
    y_pred = search.predict(X)
    assert y_pred.shape == y.shape


def test_randomized_search_cv_reproducible(binary_classification_data):
    """Test RandomizedSearchCV is reproducible with random_state."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0, 10.0, 100.0]}
    
    search1 = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=3,
        cv=2,
        random_state=42
    )
    search1.fit(X, y)
    
    search2 = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=3,
        cv=2,
        random_state=42
    )
    search2.fit(X, y)
    
    # Should get same results with same random_state
    assert search1.best_params_ == search2.best_params_


def test_grid_search_cv_multiclass(classification_data):
    """Test GridSearchCV on multiclass problem."""
    X, y = classification_data
    
    param_grid = {'C': [0.1, 1.0, 10.0]}
    
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=3)
    grid.fit(X, y)
    
    assert hasattr(grid, 'best_params_')
    y_pred = grid.predict(X)
    assert len(np.unique(y_pred)) >= 1  # At least one class predicted


def test_randomized_search_cv_multiclass(classification_data):
    """Test RandomizedSearchCV on multiclass problem."""
    X, y = classification_data
    
    param_dist = {'C': [0.1, 1.0, 10.0, 100.0]}
    
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=3,
        cv=2,
        random_state=42
    )
    search.fit(X, y)
    
    assert hasattr(search, 'best_params_')
    y_pred = search.predict(X)
    assert len(np.unique(y_pred)) > 1


def test_grid_search_cv_error_before_fit(binary_classification_data):
    """Test GridSearchCV errors when calling methods before fit."""
    X, y = binary_classification_data
    
    param_grid = {'C': [0.1, 1.0]}
    grid = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=2)
    
    with pytest.raises(ValueError):
        grid.predict(X)
    
    with pytest.raises(ValueError):
        grid.score(X, y)


def test_randomized_search_cv_error_before_fit(binary_classification_data):
    """Test RandomizedSearchCV errors when calling methods before fit."""
    X, y = binary_classification_data
    
    param_dist = {'C': [0.1, 1.0]}
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=100),
        param_dist,
        n_iter=2,
        cv=2,
        random_state=42
    )
    
    with pytest.raises(ValueError):
        search.predict(X)
    
    with pytest.raises(ValueError):
        search.score(X, y)
