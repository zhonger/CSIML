"""Main functions or classes for analysis"""

import numpy as np
import pandas as pd

from ._base import metrics2array, metrics2msg
from ._dataclass import ResData
from .common import evaluation_batch
from .io import load_pkl, write_excel


def analyze_results(results: np.array, method: str) -> list:
    """Analyze the results from total, majority and minority ways

    The results (np.array) are like below:
    ---------------------------------------------------------------
    No.  Common names                      Names in basis1
    0    majority_train_materials          Same
    1    y_majority_train                  Same
    2    y_majority_train_pred             Same
    3    majority_valition_materials       Same
    4    y_majority_validation             Same
    5    y_majority_validation_pred        Same
    6    majority_test_materials           Same
    7    y_majoirty_test                   Same
    8    y_majority_test_pred              Same
    9    minority_train_materials          minority_test_materials
    10   y_minority_train                  y_minority_test
    11   y_minority_train_pred             y_minority_test_pred
    12   minority_validation_materials     None
    13   y_minority_validation             None
    14   y_minority_validation_pred        None
    15   minority_test_materials           None
    16   y_minority_test                   None
    17   y_minority_test_pred              None
    --------------------------------------------------------------

    Args:
        results (np.array): prediction results for many iterations.
        method (str): the method name, should be one of "basis1", "basis2", "siml",
            "over", "under".

    Returns:
        list: return analysis metrics.
    """
    summarys = []
    for result in results:
        summary = []
        data = {}
        if method == "basis1":
            data["y_minority_train"] = np.zeros(5)
            data["y_minority_train_pred"] = np.zeros(5)
            data["y_minority_validation"] = np.zeros(5)
            data["y_minority_validation_pred"] = np.zeros(5)
            data["y_minority_test"] = result[10]
            data["y_minority_test_pred"] = result[11]
            data["y_train"] = result[1]
            data["y_train_pred"] = result[2]
            data["y_validation"] = result[4]
            data["y_validation_pred"] = result[5]
            data["y_test"] = np.append(result[7], result[10])
            data["y_test_pred"] = np.append(result[8], result[11])
        else:
            data["y_minority_train"] = result[10]
            data["y_minority_train_pred"] = result[11]
            data["y_minority_validation"] = result[13]
            data["y_minority_validation_pred"] = result[14]
            data["y_minority_test"] = result[16]
            data["y_minority_test_pred"] = result[17]
            data["y_train"] = np.append(result[1], result[10])
            data["y_train_pred"] = np.append(result[2], result[11])
            data["y_validation"] = np.append(result[4], result[13])
            data["y_validation_pred"] = np.append(result[5], result[14])
            data["y_test"] = np.append(result[7], result[16])
            data["y_test_pred"] = np.append(result[8], result[17])
        res_train = ResData(
            data["y_train"],
            data["y_train_pred"],
            result[1],
            result[2],
            data["y_minority_train"],
            data["y_minority_train_pred"],
        )
        res_validation = ResData(
            data["y_validation"],
            data["y_validation_pred"],
            result[4],
            result[5],
            data["y_minority_validation"],
            data["y_minority_validation_pred"],
        )
        res_test = ResData(
            data["y_test"],
            data["y_test_pred"],
            result[7],
            result[8],
            data["y_minority_test"],
            data["y_minority_test_pred"],
        )
        summary.extend(evaluation_batch(res_train))
        summary.extend(evaluation_batch(res_validation))
        summary.extend(evaluation_batch(res_test))
        summarys.append(summary)
    return summarys


def analyze_summarys(summarys: list) -> list:
    """Analysis summarys by the mean

    Args:
        summarys (list): a list including many metrics and iterations.

    Returns:
        list: return mean of summary metrics ``summarys_mean``.
    """
    summarys = pd.DataFrame(summarys)
    summarys_mean = []
    for _, col in summarys.items():
        summarys_mean.append(col.mean())
    return summarys_mean


def save_analyzed_result(
    results: pd.DataFrame,
    filename: str,
    iteration: int = 0,
) -> None:
    """Save analyzed result to file.

    Args:
        results (pd.DataFrame): analysis results.
        filename (str): filename you want to save analysis results.
        iteration (int, optional): the number of iteration. Defaults to 0.
    """
    for i, result in enumerate(results):
        result = pd.DataFrame(result)
        sheet_name = f"iteration-{(iteration*25+i)}"
        try:
            kws = {"if_sheet_exists": "replace", "sheet_name": sheet_name}
            write_excel(result, filename, "a", **kws)
        except FileExistsError:
            kws = {"sheet_name": sheet_name}
            write_excel(result, filename, "w", **kws)


class Analysis:
    """Analysis class for prediction results

    It support the analysis from original results object or files. By default, the
    filename will be required firstly. If ``filename`` is specified, ``results`` will be
    ignored. If both two parameters are not specified, it will be not allowded.

    Args:
        results (list): the results. If it's not specified, filename should be specified.
        filename (str): the results filename (the suffix should be `.pkl`). For example,
            "bandgaps.pkl".
        method (str): the model method name, like "siml", "basis1", "basis2" or others.
            Defaults to "siml".

    Attributes:
        results (list): the loaded results object.
        method (str): the model method name, like "siml", "basis1", "basis2" or others.
        summarys (list): evaluation metircs for all iterations.
        metrics (list): the average evaluation metrics for all iterations.

    Examples:
        >>> from SIML.analysis import Analysis
        >>> tasks = [
                ["468", "basis1", "3"],
                ["3895", "basis1", "3"],
            ]

        >>> for task in tasks:  # Only print the results
                filename = f"results/results_{task[0]}_{task[1]}_{task[2]}.pkl"
                method = task[1]
                a1 = Analysis(filename=filename, method=method)
                print(a1)

        >>> for task in tasks:  # Save analysis results to excel files
                filename = f"results/results_{task[0]}_{task[1]}_{task[2]}.pkl"
                method = task[1]
                a1 = Analysis(filename=filename, method=method)
                filename = f"results/results_analysis_{task[0]}_{task[1]}_{task[2]}}.xlsx"
                a1.save(filename)
    """

    def __init__(self, **kws) -> None:
        results = kws.get("results", None)
        filename = kws.get("filename", None)
        method = kws.get("method", "siml")
        if filename:
            results = load_pkl(filename)
        elif results is None:
            raise ValueError("'results' or 'filename' should be specified!")
        self.results = results
        self.method = method
        summarys = analyze_results(results, method)
        summarys_mean = analyze_summarys(summarys)
        self.summarys = summarys
        self.metrics = summarys_mean

    def save(self, filename: str) -> None:
        """Save metrics into excel file

        Args:
            filename (str): The filename you want to save data into
        """
        save_metrics(self.metrics, filename)

    def __str__(self) -> str:
        return metrics2msg(self.metrics, self.method)


def save_metrics(
    metrics: list, filename: str = "summary.xlsx", mode: str = "w"
) -> None:
    """Save metrics into excel file

    Args:
        metrics (list): metrics for results.
        filename (str, optional): filename you want to save for metrics summary.
            Defaults to "summary.xlsx".
    """
    results = metrics2array(metrics)
    header = ["Training", "Validation", "Test"]
    kws = {"sheet_name": "summary", "header": header}
    write_excel(results, filename, mode, **kws)


def save_metrics_batch(
    dataset: str = "468",
    path: str = "results",
    decimal: int = 3,
    **kws,
) -> None:
    """Save metrics with a batch for different random seeds

    Args:
        dataset (str, optional): the name of dataset. Defaults to "468".
        path (str, optional): the path of result files. Defaults to "results".
        methods (list, optional): learning methods. Defaults to ["basis1", "basis2",
            "oversampling", "undersampling", "siml"].
        random_states (list, optional): random seed list. Defaults to [1, 3, 5, 10, 15,
            20, 30, 35, 40, 45, 50].
        decimal (int, optional): the decimal place for metrics. Defaults to 3.
    """
    methods = kws.get(
        "methods", ["basis1", "basis2", "oversampling", "undersampling", "siml"]
    )
    random_states = kws.get("random_states", [1, 3, 5, 10, 15, 20, 30, 35, 40, 45, 50])
    n = 0
    for random_state in random_states:
        df = load_metrics(dataset, path, methods, random_state, decimal=decimal)
        filename = f"summary/{dataset}_{path}.xlsx"
        kws = {"sheet_name": random_state}
        if n == 0:
            write_excel(df, filename, "w", **kws)
        else:
            write_excel(df, filename, "a", **kws)
        n += 1
        print(f"Random seed {random_state} has been saved.")


def load_metrics(
    dataset: str, path: str, methods: list, random_state: int, **kws
) -> pd.DataFrame:
    """Load metrics from files

    Args:
        dataset (str): the dataset name.
        path (str): the path for saved metric files.
        methods (list): the method name.
        random_state (int): the seed for random sampling.
        decimal (int, optional): the decimal place for metrics. Defaults to 3.

    Returns:
        pd.DataFrame: return metrics.
    """
    decimal = kws.get("decimla", 3)
    ob = []
    for method in methods:
        filename = f"{path}/results_{dataset}_{random_state}_{method}.pkl"
        ob.append(Analysis(filename=filename, method=method))
    for i in range(len(methods)):
        metrics = [round(_, decimal) for _ in ob[i].metrics]
        results = metrics2array(metrics)
        results = np.array(results).T
        summary = results if i == 0 else np.vstack((summary, results))
    df = pd.DataFrame(np.array(summary))
    return df
