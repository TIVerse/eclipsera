"""Tests for core metrics."""
import numpy as np
import pytest

from eclipsera.core.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def test_accuracy_score():
    """Test accuracy score."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])
    
    acc = accuracy_score(y_true, y_pred)
    assert acc == 0.75
    
    # Test with sample weights
    sample_weight = np.array([1, 1, 2, 1])
    acc_weighted = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    assert acc_weighted == 0.6  # 3/5


def test_accuracy_score_unnormalized():
    """Test accuracy score without normalization."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])
    
    count = accuracy_score(y_true, y_pred, normalize=False)
    assert count == 3


def test_precision_recall_fscore():
    """Test precision, recall, and F-score."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    assert precision.shape == (3,)
    assert recall.shape == (3,)
    assert fscore.shape == (3,)
    assert support.shape == (3,)


def test_precision_recall_fscore_binary():
    """Test precision, recall, F-score for binary classification."""
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fscore <= 1


def test_precision_recall_fscore_macro():
    """Test macro-averaged metrics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fscore <= 1


def test_precision_score():
    """Test precision score wrapper."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    precision = precision_score(y_true, y_pred, average='binary')
    assert 0 <= precision <= 1


def test_recall_score():
    """Test recall score wrapper."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    recall = recall_score(y_true, y_pred, average='binary')
    assert 0 <= recall <= 1


def test_f1_score():
    """Test F1 score wrapper."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    f1 = f1_score(y_true, y_pred, average='binary')
    assert 0 <= f1 <= 1


def test_fbeta_score():
    """Test F-beta score with different beta values."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    # Beta = 0.5 (precision weighted more)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, average='binary')
    
    # Beta = 2.0 (recall weighted more)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, average='binary')
    
    assert 0 <= f05 <= 1
    assert 0 <= f2 <= 1


def test_confusion_matrix():
    """Test confusion matrix."""
    y_true = np.array([2, 0, 2, 2, 0, 1])
    y_pred = np.array([0, 0, 2, 2, 0, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)


def test_confusion_matrix_normalized():
    """Test normalized confusion matrix."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Each row should sum to 1
    assert np.allclose(cm_norm.sum(axis=1), 1.0)


def test_r2_score():
    """Test R² score."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    r2 = r2_score(y_true, y_pred)
    
    assert -np.inf < r2 <= 1.0


def test_r2_score_multioutput():
    """Test R² score with multiple outputs."""
    y_true = np.array([[3, -0.5], [2, 7], [1, 3]])
    y_pred = np.array([[2.5, 0.0], [2, 8], [1.5, 3.5]])
    
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    
    assert -np.inf < r2 <= 1.0


def test_r2_score_perfect():
    """Test R² score with perfect predictions."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = y_true.copy()
    
    r2 = r2_score(y_true, y_pred)
    
    assert np.isclose(r2, 1.0)


def test_mean_squared_error():
    """Test MSE."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    mse = mean_squared_error(y_true, y_pred)
    
    assert mse >= 0


def test_mean_squared_error_rmse():
    """Test RMSE."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred, squared=True)
    
    assert np.isclose(rmse, np.sqrt(mse))


def test_mean_absolute_error():
    """Test MAE."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    mae = mean_absolute_error(y_true, y_pred)
    
    assert mae >= 0
    assert mae == 0.5


def test_roc_curve():
    """Test ROC curve computation."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    assert len(fpr) == len(tpr) == len(thresholds)
    assert fpr[0] == 0.0  # First point should be (0, 0)
    assert tpr[0] == 0.0


def test_auc():
    """Test AUC computation."""
    x = np.array([0, 0.5, 1])
    y = np.array([0, 0.5, 1])
    
    area = auc(x, y)
    
    assert np.isclose(area, 0.5)


def test_roc_auc_score_binary():
    """Test ROC AUC score for binary classification."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    
    auc_score = roc_auc_score(y_true, y_score)
    
    assert 0 <= auc_score <= 1


def test_roc_auc_score_multiclass():
    """Test ROC AUC score for multiclass classification."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_score = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.1, 0.8]
    ])
    
    auc_score = roc_auc_score(y_true, y_score, multi_class='ovr')
    
    assert 0 <= auc_score <= 1
