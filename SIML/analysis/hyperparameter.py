"""Hyperparameter analysis module"""

import numpy as np
import pandas as pd

from ._base import handle_key_value
from .io import save_results, write_excel
from .main import analyze_results


def analyze_hyperparameter(op_results: np.array, op_object: str = "C") -> np.array:
    """Analyze hyperparameter optimization results

    Args:
        op_results (np.array): optimization results.
        op_object (str, optional): the optimized hyperparameter. Defaults to "C".

    Returns:
        np.array: return metrics summary ``op_summarys``.
    """
    op_summarys = []
    for op_result in op_results:
        op_summary = []
        op_obejcts = handle_key_value(op_result)
        op_summary.append(op_obejcts[op_object])
        summary = analyze_results(op_result, "op")
        metrics = np.mean(summary, axis=0)
        op_summary.append(metrics)
        op_summary.append(summary)
        op_summarys.append(op_summary)
    return op_summarys


def save_hyperparameter_metrics(
    op_results: np.array, op_object: str = "C", filename: str = "metrics.xlsx", **kws
) -> list:
    """Save metrics for optimization of hyperparameter into excel file

    Args:
        op_results (np.array): the optimization results for one hyperparameter.
        op_object (str, optional): the optimized hyperparameter. Defaults to "C".
        filename (str, optional): filename you want to save metrics. Defaults to
            "metrics.xlsx".
        is_save (str, optional): whether save metrics to excel files seperately.
            Defaults to False.

    Returns:
        list: return optimization results for hyperparameter ``op_metrics``.
    """
    metrics = [
        "mae",
        "std",
        "majority_mae",
        "majority_std",
        "minority_mae",
        "minority_std",
        "var",
        "rmse",
        "mape",
        "r2",
    ]
    op_metrics_columns = [op_object, "iteration"]
    for s in ["train", "validation", "test"]:
        for m in metrics:
            op_metrics_columns.append(f"{s}_{m}")
    is_save = kws.get("is_save", False)
    op_metrics = get_hyperparameter_metrics(op_results, op_object, is_save)
    op_metrics = pd.DataFrame(op_metrics, columns=op_metrics_columns)
    ordered_metric_mae = order_by_metric(op_metrics)
    ordered_metric_rmse = order_by_metric(op_metrics, "rmse")
    if is_save:
        write_excel(op_metrics, filename, "w", sheet_name="op_metrics")
        save_ordered_metrics(ordered_metric_mae, filename)
        save_ordered_metrics(ordered_metric_rmse, filename, "rmse")
    return op_metrics, ordered_metric_mae, ordered_metric_rmse


def save_ordered_metrics(ordered_metric: list, filename: str, metric: str = "mae"):
    write_excel(ordered_metric[0], filename, "a", sheet_name=f"by_total_{metric}")
    write_excel(ordered_metric[1], filename, "a", sheet_name=f"by_majority_{metric}")
    write_excel(ordered_metric[2], filename, "a", sheet_name=f"by_minority_{metric}")
    ordered_metric[0].insert(0, "by", f"total_{metric}")
    ordered_metric[1].insert(0, "by", f"majority_{metric}")
    ordered_metric[2].insert(0, "by", f"minority_{metric}")
    summary = pd.concat(ordered_metric)
    # summary = pd.concat([ordered_metric[0], ordered_metric[1], ordered_metric[2]])
    summary = summary.sort_values(by=["iteration", "by"], ascending=(True, True))
    write_excel(summary, filename, "a", sheet_name=f"summary_{metric}")


def order_by_metric(op_metrics: pd.DataFrame, metric: str = "mae"):
    select_columns = [
        "C",
        "iteration",
        "test_mae",
        "test_majority_mae",
        "test_minority_mae",
        "test_rmse",
        "test_majority_rmse",
        "test_minority_rmse",
    ]
    op_metrics = op_metrics.loc[:, select_columns]
    op_metrics_g = op_metrics.groupby("iteration")
    total_results = op_metrics.iloc[op_metrics_g[f"test_{metric}"].idxmin()]
    majority_results = op_metrics.iloc[op_metrics_g[f"test_majority_{metric}"].idxmin()]
    minority_results = op_metrics.iloc[op_metrics_g[f"test_minority_{metric}"].idxmin()]
    return total_results, majority_results, minority_results


def get_hyperparameter_metrics(
    op_results: np.array, op_object: str = "C", is_save: bool = False
):
    op_metrics = []
    for op_result in op_results:
        op_objects = handle_key_value(op_result)
        method = str(op_objects["method"][0])
        cv_method = str(op_objects["cv_method"][0])
        sampling_method = str(op_objects["sampling_method"][0])
        epsilon = str(op_objects["epsilon"][0])
        C = str(op_objects[op_object][0])
        data_sheet = str(op_objects["data_sheet"][0])
        metrics = []
        metrics = np.append(
            np.full([10, 1], op_objects[op_object]),
            np.arange(0, 10).reshape(-1, 1),
            axis=1,
        )
        metrics = np.append(metrics, analyze_results(op_result, "op"), axis=1)
        if len(op_metrics) == 0:
            op_metrics = metrics
        else:
            op_metrics = np.vstack((op_metrics, metrics))
        result_filename = ""
        if sampling_method == "None":
            result_filename = (
                f"{method}_{cv_method}_{epsilon}_{C}_{data_sheet}_results.xlsx"
            )
        else:
            result_filename = (
                f"{method}_{cv_method}_{sampling_method}_"
                f"{epsilon}_{C}_{data_sheet}_results.xlsx"
            )
        if is_save:
            save_results(op_result, result_filename)
    return op_metrics
