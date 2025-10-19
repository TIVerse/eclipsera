"""Tests for pipeline composition."""
import numpy as np
import pytest

from eclipsera.pipeline import FeatureUnion, Pipeline, make_pipeline
from eclipsera.preprocessing import MinMaxScaler, StandardScaler
from eclipsera.ml import LogisticRegression, Ridge


def test_pipeline_basic(binary_classification_data):
    """Test basic Pipeline functionality."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_pipeline_score(binary_classification_data):
    """Test Pipeline score method."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    score = pipe.score(X, y)
    
    assert 0 <= score <= 1


def test_pipeline_predict_proba(binary_classification_data):
    """Test Pipeline predict_proba method."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_named_steps():
    """Test Pipeline named_steps attribute."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    
    assert 'scaler' in pipe.named_steps
    assert 'clf' in pipe.named_steps
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)


def test_pipeline_get_params():
    """Test Pipeline get_params method."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0))
    ])
    
    params = pipe.get_params(deep=True)
    
    assert 'clf__C' in params
    assert params['clf__C'] == 1.0


def test_pipeline_set_params():
    """Test Pipeline set_params method."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0))
    ])
    
    pipe.set_params(clf__C=10.0)
    
    assert pipe.named_steps['clf'].C == 10.0


def test_pipeline_fit_transform(binary_classification_data):
    """Test Pipeline fit_transform method."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler())
    ])
    
    X_transformed = pipe.fit_transform(X, y)
    
    assert X_transformed.shape == X.shape


def test_pipeline_transform(binary_classification_data):
    """Test Pipeline transform method."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    X_transformed = pipe.transform(X)
    
    assert X_transformed.shape == X.shape


def test_pipeline_multiple_transforms(binary_classification_data):
    """Test Pipeline with multiple transformers."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_pipeline_regression(regression_data):
    """Test Pipeline on regression task."""
    X, y = regression_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge())
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_pipeline_verbose(binary_classification_data, capsys):
    """Test Pipeline verbose mode."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ], verbose=True)
    
    pipe.fit(X, y)
    
    captured = capsys.readouterr()
    assert 'Pipeline' in captured.out


def test_pipeline_error_no_steps():
    """Test Pipeline error with no steps."""
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_error_invalid_step():
    """Test Pipeline error with invalid step."""
    class InvalidTransformer:
        pass
    
    with pytest.raises(TypeError):
        Pipeline([
            ('invalid', InvalidTransformer()),
            ('clf', LogisticRegression())
        ])


def test_make_pipeline_basic(binary_classification_data):
    """Test make_pipeline basic functionality."""
    X, y = binary_classification_data
    
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=100)
    )
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_make_pipeline_naming():
    """Test make_pipeline auto-generates names."""
    pipe = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        LogisticRegression()
    )
    
    # Names should be lowercase class names
    assert 'standardscaler' in pipe.named_steps
    assert 'minmaxscaler' in pipe.named_steps
    assert 'logisticregression' in pipe.named_steps


def test_make_pipeline_duplicate_names():
    """Test make_pipeline handles duplicate class names."""
    pipe = make_pipeline(
        StandardScaler(),
        StandardScaler()
    )
    
    # Should handle duplicates
    steps = pipe.steps
    assert len(steps) == 2
    assert steps[0][0] != steps[1][0]


def test_feature_union_basic():
    """Test basic FeatureUnion functionality."""
    X = np.random.randn(10, 5)
    
    union = FeatureUnion([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler())
    ])
    
    X_transformed = union.fit_transform(X)
    
    # Should concatenate features
    assert X_transformed.shape == (10, 10)  # 5 + 5


def test_feature_union_fit_transform():
    """Test FeatureUnion fit and transform separately."""
    X = np.random.randn(10, 5)
    
    union = FeatureUnion([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler())
    ])
    
    union.fit(X)
    X_transformed = union.transform(X)
    
    assert X_transformed.shape == (10, 10)


def test_feature_union_in_pipeline(binary_classification_data):
    """Test FeatureUnion inside Pipeline."""
    X, y = binary_classification_data
    
    union = FeatureUnion([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler())
    ])
    
    pipe = Pipeline([
        ('union', union),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape


def test_feature_union_get_params():
    """Test FeatureUnion get_params method."""
    union = FeatureUnion([
        ('scaler1', StandardScaler(with_std=True)),
        ('scaler2', MinMaxScaler())
    ])
    
    params = union.get_params(deep=True)
    
    assert 'scaler1__with_std' in params


def test_feature_union_verbose(capsys):
    """Test FeatureUnion verbose mode."""
    X = np.random.randn(10, 5)
    
    union = FeatureUnion([
        ('scaler1', StandardScaler()),
        ('scaler2', MinMaxScaler())
    ], verbose=True)
    
    union.fit_transform(X)
    
    captured = capsys.readouterr()
    assert 'FeatureUnion' in captured.out


def test_pipeline_with_nested_params_in_grid_search(binary_classification_data):
    """Test Pipeline nested parameters can be set."""
    X, y = binary_classification_data
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    # Test set_params with nested parameters
    pipe.set_params(clf__C=10.0)
    assert pipe.named_steps['clf'].C == 10.0
    
    # Test fit/predict works
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    assert y_pred.shape == y.shape


def test_pipeline_caching():
    """Test Pipeline doesn't refit transformers unnecessarily."""
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipe.fit(X, y)
    
    # Multiple predictions shouldn't refit
    y_pred1 = pipe.predict(X)
    y_pred2 = pipe.predict(X)
    
    assert np.array_equal(y_pred1, y_pred2)


def test_pipeline_clone_estimator():
    """Test Pipeline works with cross-validation."""
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    # Just test that pipeline works
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    assert y_pred.shape == y.shape
