"""Metrics for model evaluation in Eclipsera.

This module provides a comprehensive set of metrics for classification, regression,
clustering, and ranking tasks.
"""
from typing import Any, Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
from scipy.special import xlogy

from .validation import check_array, check_consistent_length, column_or_1d, type_of_target


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Accuracy classification score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    normalize : bool, default=True
        If False, return number of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        Accuracy score.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 3])
    >>> y_pred = np.array([0, 1, 1, 3])
    >>> accuracy_score(y_true, y_pred)
    0.75
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)
    
    if sample_weight is None:
        score = np.sum(y_true == y_pred)
    else:
        score = np.sum(sample_weight[y_true == y_pred])
    
    if normalize:
        n_samples = len(y_true) if sample_weight is None else np.sum(sample_weight)
        return score / n_samples
    else:
        return score


def precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
    labels: Optional[np.ndarray] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    zero_division: Union[str, float] = "warn",
) -> tuple:
    """Compute precision, recall, F-score and support.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    beta : float, default=1.0
        F-beta score beta parameter.
    labels : array-like, default=None
        The set of labels to include.
    pos_label : str or int, default=1
        The class to report for binary classification.
    average : {'micro', 'macro', 'weighted', 'binary'} or None, default=None
        Type of averaging to perform.
    sample_weight : array-like, default=None
        Sample weights.
    zero_division : 'warn', 0 or 1, default='warn'
        Value to return when there is a zero division.

    Returns
    -------
    precision : float or ndarray
    recall : float or ndarray
    fbeta_score : float or ndarray
    support : int or ndarray
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    
    # Compute true positives, false positives, false negatives
    tp = np.zeros(n_labels, dtype=np.float64)
    fp = np.zeros(n_labels, dtype=np.float64)
    fn = np.zeros(n_labels, dtype=np.float64)
    support = np.zeros(n_labels, dtype=np.int64)
    
    for i, label in enumerate(labels):
        true_mask = y_true == label
        pred_mask = y_pred == label
        
        if sample_weight is None:
            tp[i] = np.sum(true_mask & pred_mask)
            fp[i] = np.sum(~true_mask & pred_mask)
            fn[i] = np.sum(true_mask & ~pred_mask)
            support[i] = np.sum(true_mask)
        else:
            tp[i] = np.sum(sample_weight[true_mask & pred_mask])
            fp[i] = np.sum(sample_weight[~true_mask & pred_mask])
            fn[i] = np.sum(sample_weight[true_mask & ~pred_mask])
            support[i] = np.sum(sample_weight[true_mask])
    
    # Compute precision, recall, f-score
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        if beta == 0:
            f_score = precision
        else:
            f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    # Handle zero division
    if zero_division == "warn":
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(recall)] = 0.0
        f_score[np.isnan(f_score)] = 0.0
    elif isinstance(zero_division, (int, float)):
        precision[np.isnan(precision)] = zero_division
        recall[np.isnan(recall)] = zero_division
        f_score[np.isnan(f_score)] = zero_division
    
    # Apply averaging
    if average == "micro":
        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)
        fn_sum = np.sum(fn)
        
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        
        if beta == 0:
            f_score = precision
        else:
            f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
        
        support = np.sum(support)
    
    elif average == "macro":
        precision = np.mean(precision)
        recall = np.mean(recall)
        f_score = np.mean(f_score)
        support = None
    
    elif average == "weighted":
        weights = support / np.sum(support)
        precision = np.sum(precision * weights)
        recall = np.sum(recall * weights)
        f_score = np.sum(f_score * weights)
        support = np.sum(support)
    
    elif average == "binary":
        if n_labels != 2:
            raise ValueError("binary averaging requires exactly 2 labels")
        pos_idx = np.where(labels == pos_label)[0][0]
        precision = precision[pos_idx]
        recall = recall[pos_idx]
        f_score = f_score[pos_idx]
        support = support[pos_idx]
    
    return precision, recall, f_score, support


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Compute precision score."""
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return p


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Compute recall score."""
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return r


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Compute F1 score."""
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return f


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float, **kwargs: Any) -> float:
    """Compute F-beta score."""
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, **kwargs)
    return f


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, default=None
        List of labels to index the matrix.
    sample_weight : array-like, default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalization mode.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}
    
    # Build confusion matrix
    CM = np.zeros((n_labels, n_labels), dtype=np.int64 if sample_weight is None else np.float64)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_ind and pred_label in label_to_ind:
            i = label_to_ind[true_label]
            j = label_to_ind[pred_label]
            if sample_weight is None:
                CM[i, j] += 1
            else:
                idx = np.where((y_true == true_label) & (y_pred == pred_label))[0][0]
                CM[i, j] += sample_weight[idx]
    
    # Normalize if requested
    if normalize == "true":
        CM = CM / CM.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        CM = CM / CM.sum(axis=0, keepdims=True)
    elif normalize == "all":
        CM = CM / CM.sum()
    
    return CM


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """R^2 (coefficient of determination) score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, default='uniform_average'
        Defines aggregating of multiple output scores.

    Returns
    -------
    score : float or ndarray
        R^2 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        weight = sample_weight[:, np.newaxis] if y_true.ndim == 2 else sample_weight
    else:
        weight = 1.0
    
    numerator = np.sum(weight * (y_true - y_pred) ** 2, axis=0)
    denominator = np.sum(weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2, axis=0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        score = 1 - (numerator / denominator)
    
    # Handle edge cases
    if np.isscalar(score):
        if denominator == 0:
            score = 0.0
    else:
        score[denominator == 0] = 0.0
    
    if multioutput == "raw_values":
        return score
    elif multioutput == "uniform_average":
        return np.mean(score)
    elif multioutput == "variance_weighted":
        return np.average(score, weights=denominator)


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    squared: bool = True,
) -> float:
    """Mean squared error (MSE) or root mean squared error (RMSE).

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    sample_weight : array-like, default=None
        Sample weights.
    squared : bool, default=True
        If True, returns MSE. If False, returns RMSE.

    Returns
    -------
    loss : float
        MSE or RMSE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    
    mse = np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)
    
    if squared:
        return np.mean(mse) if mse.ndim > 0 else float(mse)
    else:
        return np.mean(np.sqrt(mse)) if mse.ndim > 0 else float(np.sqrt(mse))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Mean absolute error (MAE).

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    sample_weight : array-like, default=None
        Sample weights.

    Returns
    -------
    loss : float
        MAE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    
    mae = np.average(np.abs(y_true - y_pred), axis=0, weights=sample_weight)
    
    return np.mean(mae) if mae.ndim > 0 else float(mae)


def roc_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    sample_weight: Optional[np.ndarray] = None,
    multi_class: str = "raise",
) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Target scores.
    average : {'micro', 'macro', 'weighted'}, default='macro'
        Type of averaging.
    sample_weight : array-like, default=None
        Sample weights.
    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
        Multi-class strategy.

    Returns
    -------
    auc : float
        ROC AUC score.
    """
    y_true = column_or_1d(y_true)
    y_score = np.asarray(y_score)
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if n_classes == 2:
        # Binary classification
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=classes[1], sample_weight=sample_weight)
        return auc(fpr, tpr)
    else:
        # Multiclass
        if multi_class == "raise":
            raise ValueError("multi_class must be 'ovr' or 'ovo' for multiclass problems")
        
        # Simple approximation for multiclass
        if y_score.ndim == 1:
            raise ValueError("y_score must be 2D for multiclass problems")
        
        aucs = []
        for i, c in enumerate(classes):
            y_true_binary = (y_true == c).astype(int)
            y_score_binary = y_score[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary, sample_weight=sample_weight)
            aucs.append(auc(fpr, tpr))
        
        if average == "macro":
            return np.mean(aucs)
        elif average == "weighted":
            weights = np.array([np.sum(y_true == c) for c in classes])
            return np.average(aucs, weights=weights)
        else:
            return np.mean(aucs)


def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: Optional[Union[int, str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
) -> tuple:
    """Compute Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores.
    pos_label : int or str, default=None
        Label of positive class.
    sample_weight : array-like, default=None
        Sample weights.
    drop_intermediate : bool, default=True
        Whether to drop suboptimal thresholds.

    Returns
    -------
    fpr : ndarray
        False positive rates.
    tpr : ndarray
        True positive rates.
    thresholds : ndarray
        Decreasing thresholds on the decision function.
    """
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    
    if pos_label is None:
        pos_label = 1
    
    # Make y_true binary
    y_true = (y_true == pos_label).astype(int)
    
    # Sort by score
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]
    else:
        sample_weight = np.ones_like(y_true)
    
    # Compute TPR and FPR
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
    fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]
    
    thresholds = y_score[threshold_idxs]
    
    # Add endpoint
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    
    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]
    
    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    
    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Parameters
    ----------
    x : ndarray
        x coordinates.
    y : ndarray
        y coordinates.

    Returns
    -------
    auc : float
        Area under the curve.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    direction = 1
    if x[0] > x[-1]:
        # Reverse for decreasing x
        x = x[::-1]
        y = y[::-1]
        direction = -1
    
    area = np.trapezoid(y, x) * direction
    return float(area)


class MetricRegistry:
    """Registry for metrics with metadata."""
    
    _metrics = {
        "accuracy": {"func": accuracy_score, "type": "classification", "requires_proba": False},
        "precision": {"func": precision_score, "type": "classification", "requires_proba": False},
        "recall": {"func": recall_score, "type": "classification", "requires_proba": False},
        "f1": {"func": f1_score, "type": "classification", "requires_proba": False},
        "roc_auc": {"func": roc_auc_score, "type": "classification", "requires_proba": True},
        "r2": {"func": r2_score, "type": "regression", "requires_proba": False},
        "mse": {"func": mean_squared_error, "type": "regression", "requires_proba": False},
        "mae": {"func": mean_absolute_error, "type": "regression", "requires_proba": False},
    }
    
    @classmethod
    def get(cls, name: str) -> dict:
        """Get metric by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return cls._metrics[name]
    
    @classmethod
    def list_metrics(cls, metric_type: Optional[str] = None) -> list:
        """List available metrics."""
        if metric_type is None:
            return list(cls._metrics.keys())
        return [name for name, info in cls._metrics.items() if info["type"] == metric_type]
