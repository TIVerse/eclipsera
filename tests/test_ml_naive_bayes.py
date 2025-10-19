"""Tests for Naive Bayes algorithms."""
import numpy as np
import pytest

from eclipsera.ml.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


def test_gaussian_nb_fit_predict(classification_data):
    """Test GaussianNB fit and predict."""
    X, y = classification_data
    
    clf = GaussianNB()
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'theta_')
    assert hasattr(clf, 'var_')
    assert hasattr(clf, 'class_prior_')
    assert hasattr(clf, 'classes_')
    
    # Check shapes
    assert clf.theta_.shape == (clf.n_classes_, X.shape[1])
    assert clf.var_.shape == (clf.n_classes_, X.shape[1])
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy is reasonable
    accuracy = clf.score(X, y)
    assert accuracy > 0.3


def test_gaussian_nb_predict_proba(binary_classification_data):
    """Test GaussianNB predict_proba."""
    X, y = binary_classification_data
    
    clf = GaussianNB()
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_gaussian_nb_var_smoothing():
    """Test GaussianNB with different var_smoothing."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = GaussianNB(var_smoothing=1e-5)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_gaussian_nb_small_dataset():
    """Test GaussianNB on small dataset."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = GaussianNB()
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)


def test_multinomial_nb_fit_predict():
    """Test MultinomialNB fit and predict."""
    # Count data (e.g., word counts)
    X = np.array([
        [1, 0, 2, 0],
        [0, 1, 0, 3],
        [2, 0, 1, 0],
        [0, 2, 0, 1]
    ])
    y = np.array([0, 1, 0, 1])
    
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'feature_log_prob_')
    assert hasattr(clf, 'class_log_prior_')
    assert hasattr(clf, 'classes_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check probability predictions
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multinomial_nb_alpha_smoothing():
    """Test MultinomialNB with different alpha values."""
    X = np.array([[1, 2], [2, 1], [3, 0], [0, 3]])
    y = np.array([0, 0, 1, 1])
    
    # Test with alpha=0 (no smoothing)
    clf = MultinomialNB(alpha=0.0)
    clf.fit(X, y)
    y_pred1 = clf.predict(X)
    
    # Test with alpha=1.0 (Laplace smoothing)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, y)
    y_pred2 = clf.predict(X)
    
    assert y_pred1.shape == y.shape
    assert y_pred2.shape == y.shape


def test_multinomial_nb_no_fit_prior():
    """Test MultinomialNB without fitting prior."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1])
    
    clf = MultinomialNB(fit_prior=False)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_multinomial_nb_negative_values_error():
    """Test MultinomialNB raises error for negative values."""
    X = np.array([[1, -1], [2, 3]])
    y = np.array([0, 1])
    
    clf = MultinomialNB()
    
    with pytest.raises(ValueError, match="Negative values"):
        clf.fit(X, y)


def test_bernoulli_nb_fit_predict():
    """Test BernoulliNB fit and predict."""
    # Binary features
    X = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ])
    y = np.array([0, 1, 0, 1])
    
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(clf, 'feature_log_prob_')
    assert hasattr(clf, 'feature_log_prob_neg_')
    assert hasattr(clf, 'class_log_prior_')
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check probability predictions
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bernoulli_nb_binarize():
    """Test BernoulliNB with binarization."""
    # Continuous data that needs binarization
    X = np.array([
        [0.1, 0.9, 0.2, 0.8],
        [0.8, 0.2, 0.7, 0.1],
        [0.3, 0.1, 0.9, 0.7],
    ])
    y = np.array([0, 1, 0])
    
    clf = BernoulliNB(binarize=0.5)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_bernoulli_nb_no_binarize():
    """Test BernoulliNB without binarization."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1])
    
    clf = BernoulliNB(binarize=None)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_bernoulli_nb_alpha_smoothing():
    """Test BernoulliNB with different alpha values."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1])
    
    # Test with different alpha values
    for alpha in [0.1, 1.0, 2.0]:
        clf = BernoulliNB(alpha=alpha)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


def test_bernoulli_nb_no_fit_prior():
    """Test BernoulliNB without fitting prior."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1])
    
    clf = BernoulliNB(fit_prior=False)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_naive_bayes_multiclass(classification_data):
    """Test all Naive Bayes variants on multiclass problem."""
    X, y = classification_data
    
    # GaussianNB
    clf = GaussianNB()
    clf.fit(X, y)
    assert len(clf.classes_) == 3
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # MultinomialNB (make data non-negative)
    X_pos = np.abs(X)
    clf = MultinomialNB()
    clf.fit(X_pos, y)
    assert len(clf.classes_) == 3
    
    # BernoulliNB
    clf = BernoulliNB()
    clf.fit(X, y)
    assert len(clf.classes_) == 3


def test_gaussian_nb_log_proba():
    """Test GaussianNB predict_log_proba."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = GaussianNB()
    clf.fit(X, y)
    
    log_proba = clf.predict_log_proba(X)
    proba = clf.predict_proba(X)
    
    # log_proba should be log of proba
    assert np.allclose(np.exp(log_proba), proba, atol=1e-10)


def test_multinomial_nb_log_proba():
    """Test MultinomialNB predict_log_proba."""
    X = np.array([[1, 2], [2, 1], [3, 0], [0, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = MultinomialNB()
    clf.fit(X, y)
    
    log_proba = clf.predict_log_proba(X)
    proba = clf.predict_proba(X)
    
    assert np.allclose(np.exp(log_proba), proba, atol=1e-10)


def test_bernoulli_nb_log_proba():
    """Test BernoulliNB predict_log_proba."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 0, 1])
    
    clf = BernoulliNB()
    clf.fit(X, y)
    
    log_proba = clf.predict_log_proba(X)
    proba = clf.predict_proba(X)
    
    assert np.allclose(np.exp(log_proba), proba, atol=1e-10)
