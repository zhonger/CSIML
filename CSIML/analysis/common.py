"""Common functions for analysis"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from ._dataclass import ResData


def evaluation(y: np.array, y_pred: np.array) -> dict:
    """Calculate metrics from experimental and prediced y

    Args:
        y (np.array): experimental y
        y_pred (np.array): predicted y

    Returns:
        dict: return metrics dict.
    """
    # error = y_pred - y
    error = abs(y - y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    std = np.std(y_pred - y)
    metrics = {
        "error": error,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "std": std,
    }
    return metrics


def evaluation_metric(y: np.array, y_pred: np.array, metric: str) -> float:
    """Evaluate performance in term of one specified metric

    Args:
        y (np.array): Experimental values.
        y_pred (np.array): Prediction values.
        metric (str): Evaluation metric, now supporting "mae", "mae_std", "rmse",
            "variance", "mape", "r2".

    Returns:
        float: The evaluation metric value.
    """
    METRIC = {
        "mae": mean_absolute_error(y, y_pred),
        "mae_std": np.std(y_pred - y),
        "mse": mean_squared_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y, y_pred),
        "r2": r2_score(y, y_pred),
        "variance": median_absolute_error(y, y_pred),
    }
    return METRIC.get(metric)


def evaluation_metrics(y: np.array, y_pred: np.array, metrics: list) -> list:
    """Evaluate performance in term of specified metrics

    Args:
        y (np.array): Experimental values.
        y_pred (np.array): Prediction values.
        metrics (list): Evaluation metrics, should be a list of supporting metrics.

    Returns:
        list: The evaluation metric values.
    """
    metric_values = []
    for metric in metrics:
        metric_values.append(evaluation_metric(y, y_pred, metric))
    return metric_values


def evaluation_batch(res: ResData) -> list:
    """Evaluation performance for different results and metrics as tasks

    Args:
        res (ResData): result dataset including six lists.

    Returns:
        list: The evaluation metric values.
    """
    tasks = [
        ["y", "y_pred", "mae"],
        ["y", "y_pred", "mae_std"],
        ["y_majority", "y_majority_pred", "mae"],
        ["y_majority", "y_majority_pred", "mae_std"],
        ["y_minority", "y_minority_pred", "mae"],
        ["y_minority", "y_minority_pred", "mae_std"],
        ["y", "y_pred", "variance"],
        ["y", "y_pred", "rmse"],
        ["y_majority", "y_majority_pred", "rmse"],
        ["y_minority", "y_minority_pred", "rmse"],
        ["y", "y_pred", "mape"],
        ["y", "y_pred", "r2"],
    ]
    metric_values = []
    for task in tasks:
        y = getattr(res, task[0])
        y_pred = getattr(res, task[1])
        metric = task[2]
        metric_value = evaluation_metric(y, y_pred, metric)
        metric_values.append(metric_value)
    return metric_values
