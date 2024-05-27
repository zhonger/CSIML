"""Hyperparameter analysis module"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from CSIML.analysis._base import handle_key_value
from CSIML.analysis.io import save_results, write_excel, write_pkl
from CSIML.analysis.main import Analysis, analyze_results
from CSIML.model.iml import IML, Basis2Model


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


def save_ordered_metrics(
    ordered_metric: list, filename: str, metric: str = "mae"
) -> None:
    """Save ordered metrics to files.

    Args:
        ordered_metric (list): ordered metrics.
        filename (str): the name of the file to save.
        metric (str, optional): the specific metric to order. Defaults to "mae".
    """
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


def order_by_metric(
    op_metrics: pd.DataFrame, metric: str = "mae"
) -> tuple[list, list, list]:
    """Order metrics by one specific metric

    Args:
        op_metrics (pd.DataFrame): metrics for hyperparameter optimization.
        metric (str, optional): the specific metric to order. Defaults to "mae".

    Returns:
        tuple[list, list, list]: return ordered metric in total, majority and minority set.
    """
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
) -> list:
    """Get metrics for hyperparameter optimization.

    Args:
        op_results (np.array): the hyperparameter optimization results.
        op_object (str, optional): the optmization hyperparameter name. Defaults to "C".
        is_save (bool, optional): whether to save to the file. Defaults to False.

    Returns:
        list: return the metrics for hyperparameter optimization.
    """
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


def cal_single_hyper(
    para: tuple, path: str, method: str = "basis2", **kws
) -> None:
    """Calculate for single hyper

    Args:
        para (tuple): the hyperparameters values.
        path (str): the path to save calculated result.
        data (pd.DataFrame)): the dataset.
        method (str, optional): the method name. Defaults to "basis2".
        hyper_names (list, optional): the names of hyperparameters. Defaults to
            ["degree", "C].
        cv_method (str, optional): the cross-validation method name. Defaults to "siml
        threshold (float, optional): the threshold for splitting majority and minority
            set. Defaults to 5.0.
        basic_model (str, optional): the basic ML model. Defaults to "pSVR".

    Raises:
        ValueError: when the data is not defined or the parameters are illegal.
    """
    hyper_names = kws.get("hyper_names", ["degree", "C"])
    cv_method = kws.get("cv_method", "siml")
    if "data" not in kws:
        raise ValueError("No data specified in the parameters.")
    data = kws.get("data")
    threshold = kws.get("threshold", 5)
    basic_model = kws.get("basic_model", "pSVR")
    if len(para) != len(hyper_names):
        raise ValueError("The length of `hyper_names` should be equal to parameters.")
    parameters = {}
    for p, h in zip(para, hyper_names):
        parameters[h] = p
    if method in ["basis2", "Basis2"]:
        model = Basis2Model(
            data,
            threshold=threshold,
            normalize=True,
            basic_model=basic_model,
            parameters=parameters,
        )
        path = f"{path}/{basic_model}/dC{para[0]}"
    elif method in ["oversampling", "Oversampling"]:
        model = Basis2Model(
            data,
            threshold=threshold,
            normalize=True,
            sampling_method="oversampling",
            basic_model=basic_model,
            parameters=parameters,
        )
        path = f"{path}/{basic_model}/overdC{para[0]}"
    else:
        model = IML(
            data,
            threshold=threshold,
            normalize=True,
            basic_model=basic_model,
            parameters=parameters,
            opt_C=False,
            n_jobs=1,
            **kws,
        )
        path = f"{path}/{basic_model}/SIMLdC{para[0]}"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f"{path}/{method}_{cv_method}_{para[-1]}.pkl"
    results = model.fit_predict()
    write_pkl(results, filename)
    if kws.get("show_tips", False):
        for p, h in zip(para, hyper_names):
            print(f"{h}: {p}", end=" ")
        print("(Finished)")


def cal_hypers(
    degrees: list,
    Cs: list,
    path: str,
    method: str = "basis2",
    **kws,
) -> None:
    """Calculate for hyperparameters

    Args:
        degrees (list): the degrees.
        Cs (list): the hyperparameter C list.
        path (str): the path to save result files.
        data (pd.DataFrame): the dataset.
        method (str, optional): the method name. Defaults to "basis2".
        n_jobs (int, optional): the cpu cors used. Defaults to the length of 
            hyperparameter list.

    Raises:
        ValueError: when data is not defined.
    """
    if "data" not in kws:
        raise ValueError("No data specified in the parameters.")
    paras = []
    for degree in degrees:
        for C in Cs:
            paras.append((degree, C))
    n_jobs = kws.get("n_jobs", len(paras))
    Parallel(n_jobs=n_jobs)(
        delayed(cal_single_hyper)(para, path, method, **kws) for para in paras
    )


def load_single_hyper(para: tuple, path: str, method: str = "basis2", **kws) -> list:
    """Load single hyperparameter optimization result

    Args:
        para (tuple): the hyperparameters.
        path (str): the path to save files.
        method (str, optional): the method name. Defaults to "basis2".

    Returns:
        list: return the metrics for single hyperparameter optimization.
    """
    hyper_names = kws.get("hyper_names", ["degree", "C"])
    cv_method = kws.get("cv_method", "siml")
    basic_model = kws.get("basic_model", "pSVR")
    if method in ["basis2", "Basis2"]:
        path = f"{path}/{basic_model}/dC{para[0]}"
    elif method in ["oversampling", "Oversampling"]:
        path = f"{path}/{basic_model}/overdC{para[0]}"
    else:
        path = f"{path}/{basic_model}/SIMLdC{para[0]}"
    filename = f"{path}/{method}_{cv_method}_{para[-1]}.pkl"
    a1 = Analysis(filename=filename, method=method)
    metrics = a1.metrics
    for p in para:
        metrics.append(p)
    if kws.get("show_tips", False):
        for p, h in zip(para, hyper_names):
            print(f"{h}: {p}", end=" ")
        print("(Finished)")
    return metrics


def load_hypers(
    degrees: list,
    Cs: list,
    path: str,
    method: str = "basis2",
    **kws,
) -> list:
    """Load hyperparameter optimization results

    Args:
        degrees (list): the degrees.
        Cs (list): the hyperparameter C list.
        path (str): the path to save files.
        method (str, optional): the method name. Defaults to "basis2".

    Returns:
        list: return the metrics for hyperparameter optimizations.
    """
    paras = []
    for degree in degrees:
        for C in Cs:
            paras.append((degree, C))
    n_jobs = kws.get("n_jobs", len(paras))
    azs = Parallel(n_jobs=n_jobs)(
        delayed(load_single_hyper)(para, path, method, **kws) for para in paras
    )
    return azs
