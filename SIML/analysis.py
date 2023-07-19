import os.path
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from tqdm import tqdm


def evaluation(y: np.array, y_pred: np.array) -> dict:
    """Calculate metrics from experimental and prediced y

    Args:
        y (np.array): experimental y
        y_pred (np.array): predicted y

    Returns:
        dict: return metrics dict.

    """
    error = abs(y - y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    std = np.std(y - y_pred)
    metrics = {
        "error": error,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "std": std,
    }
    return metrics


def analyze_results(results: np.array, method: str) -> np.array:
    """Analyze the results from total, majority and minority ways

    Args:
        results (np.array): prediction results for many iterations.
        method (str): the method name, should be one of "basis1", "basis2" and "siml".

    Returns:
        np.array: return analysis metrics.

    """
    summarys = []
    for result in results:
        summary = []

        y_majority_train = result[1]
        y_majority_train_pred = result[2]
        y_majority_validation = result[4]
        y_majority_validation_pred = result[5]
        y_majority_test = result[7]
        y_majority_test_pred = result[8]

        match method:
            case "basis1":
                y_minority_train = np.zeros(5)
                y_minority_train_pred = np.zeros(5)
                y_minority_validation = np.zeros(5)
                y_minority_validation_pred = np.zeros(5)
                y_minority_test = result[10]
                y_minority_test_pred = result[11]
                y_train = y_majority_train
                y_train_pred = y_majority_train_pred
                y_validation = y_majority_validation
                y_validation_pred = y_majority_validation_pred
                y_test = np.append(y_majority_test, y_minority_test)
                y_test_pred = np.append(y_majority_test_pred, y_minority_test_pred)

            case _:
                y_minority_train = result[10]
                y_minority_train_pred = result[11]
                y_minority_validation = result[13]
                y_minority_validation_pred = result[14]
                y_minority_test = result[16]
                y_minority_test_pred = result[17]
                y_train = np.append(y_majority_train, y_minority_train)
                y_train_pred = np.append(y_majority_train_pred, y_minority_train_pred)
                y_validation = np.append(y_majority_validation, y_minority_validation)
                y_validation_pred = np.append(
                    y_majority_validation_pred, y_minority_validation_pred
                )
                y_test = np.append(y_majority_test, y_minority_test)
                y_test_pred = np.append(y_majority_test_pred, y_minority_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_std = np.std(abs(y_train - y_train_pred))
        train_majority_mae = mean_absolute_error(
            y_majority_train, y_majority_train_pred
        )
        train_majority_std = np.std(abs(y_majority_train - y_majority_train_pred))
        train_minority_mae = mean_absolute_error(
            y_minority_train, y_minority_train_pred
        )
        train_minority_std = np.std(abs(y_minority_train - y_minority_train_pred))
        train_var = median_absolute_error(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        v_mae = mean_absolute_error(y_validation, y_validation_pred)
        v_std = np.std(abs(y_validation - y_validation_pred))
        v_majority_mae = mean_absolute_error(
            y_majority_validation, y_majority_validation_pred
        )
        v_majority_std = np.std(abs(y_majority_validation - y_majority_validation_pred))
        v_minority_mae = mean_absolute_error(
            y_minority_validation, y_minority_validation_pred
        )
        v_minority_std = np.std(abs(y_minority_validation - y_minority_validation_pred))
        v_var = median_absolute_error(y_validation, y_validation_pred)
        v_rmse = mean_squared_error(y_validation, y_validation_pred)
        v_mape = mean_absolute_percentage_error(y_validation, y_validation_pred)
        v_r2 = r2_score(y_validation, y_validation_pred)
        t_mae = mean_absolute_error(y_test, y_test_pred)
        t_std = np.std(abs(y_test - y_test_pred))
        t_majority_mae = mean_absolute_error(y_majority_test, y_majority_test_pred)
        t_majority_std = np.std(abs(y_majority_test - y_majority_test_pred))
        t_minority_mae = mean_absolute_error(y_minority_test, y_minority_test_pred)
        t_minority_std = np.std(abs(y_minority_test - y_minority_test_pred))
        t_var = median_absolute_error(y_test, y_test_pred)
        t_rmse = mean_squared_error(y_test, y_test_pred)
        t_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        t_r2 = r2_score(y_test, y_test_pred)
        train_majority_rmse = mean_squared_error(
            y_majority_train, y_majority_train_pred
        )
        train_minority_rmse = mean_squared_error(
            y_minority_train, y_minority_train_pred
        )
        v_majority_rmse = mean_squared_error(
            y_majority_validation, y_majority_validation_pred
        )
        v_minority_rmse = mean_squared_error(
            y_minority_validation, y_minority_validation_pred
        )
        t_majority_rmse = mean_squared_error(y_majority_test, y_majority_test_pred)
        t_minority_rmse = mean_squared_error(y_minority_test, y_minority_test_pred)

        summary.append(train_mae)
        summary.append(train_std)
        summary.append(train_majority_mae)
        summary.append(train_majority_std)
        summary.append(train_minority_mae)
        summary.append(train_minority_std)
        summary.append(train_var)
        summary.append(train_rmse)
        summary.append(train_mape)
        summary.append(train_r2)
        summary.append(v_mae)
        summary.append(v_std)
        summary.append(v_majority_mae)
        summary.append(v_majority_std)
        summary.append(v_minority_mae)
        summary.append(v_minority_std)
        summary.append(v_var)
        summary.append(v_rmse)
        summary.append(v_mape)
        summary.append(v_r2)
        summary.append(t_mae)
        summary.append(t_std)
        summary.append(t_majority_mae)
        summary.append(t_majority_std)
        summary.append(t_minority_mae)
        summary.append(t_minority_std)
        summary.append(t_var)
        summary.append(t_rmse)
        summary.append(t_mape)
        summary.append(t_r2)
        summary.append(train_majority_rmse)
        summary.append(train_minority_rmse)
        summary.append(v_majority_rmse)
        summary.append(v_minority_rmse)
        summary.append(t_majority_rmse)
        summary.append(t_minority_rmse)

        summarys.append(summary)

    return summarys


def save_hyperparameter_metrics(
    op_results: np.array,
    op_object: str = "C",
    filename: str = "metrics.xlsx",
    save_result: bool = True,
) -> list:
    """Save metrics for optimization of hyperparameter into excel file

    Args:
        op_results (np.array): the optimization results for one hyperparameter.
        op_object (str, optional): the optimized hyperparameter. Defaults to "C".
        filename (str, optional): filename you want to save metrics. Defaults to
            "metrics.xlsx".
        save_result (bool, optional): whether saving or not. Defaults to True.

    Returns:
        list: return optimization results for hyperparameter ``op_metrics``.
    """
    op_metrics = []
    for op_result in op_results:
        result_filename = ""
        op_objects = handle_key_value(op_result)
        method = str(op_objects["method"][0])
        cv_method = str(op_objects["cv_method"][0])
        sampling_method = str(op_objects["sampling_method"][0])
        epsilon = str(op_objects["epsilon"][0])
        C = str(op_objects["C"][0])
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

        if sampling_method == "None":
            result_filename = (
                f"{method}_{cv_method}_{epsilon}_{C}_{data_sheet}_results.xlsx"
            )
        else:
            result_filename = (
                f"{method}_{cv_method}_{sampling_method}_"
                f"{epsilon}_{C}_{data_sheet}_results.xlsx"
            )

        if save_result:
            save_results(op_result, result_filename)

    op_metrics_columns = [
        op_object,
        "iteration",
        "train_mae",
        "train_std",
        "train_majority_mae",
        "train_majority_std",
        "train_minority_mae",
        "train_minority_std",
        "train_var",
        "train_rmse",
        "train_mape",
        "train_r2",
        "validation_mae",
        "validation_std",
        "validation_majority_mae",
        "validation_majority_std",
        "validation_minority_mae",
        "validation_minority_std",
        "validation_var",
        "validation_rmse",
        "validation_mape",
        "validation_r2",
        "test_mae",
        "test_std",
        "test_majority_mae",
        "test_majority_std",
        "test_minority_mae",
        "test_minority_std",
        "test_var",
        "test_rmse",
        "test_mape",
        "test_r2",
        "train_majority_rmse",
        "train_minority_rmse",
        "validation_majority_rmse",
        "validation_minority_rmse",
        "test_majority_rmse",
        "test_minority_rmse",
    ]
    op_metrics = pd.DataFrame(op_metrics, columns=op_metrics_columns)

    # By MAE
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
    total_mae_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_mae"].idxmin()
    ].loc[:, select_columns]
    majority_mae_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_majority_mae"].idxmin()
    ].loc[:, select_columns]
    minority_mae_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_minority_mae"].idxmin()
    ].loc[:, select_columns]

    # By RMSE
    select_columns = [
        "C",
        "iteration",
        "test_rmse",
        "test_majority_rmse",
        "test_minority_rmse",
        "test_mae",
        "test_majority_mae",
        "test_minority_mae",
    ]
    total_rmse_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_rmse"].idxmin()
    ].loc[:, select_columns]
    majority_rmse_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_majority_rmse"].idxmin()
    ].loc[:, select_columns]
    minority_rmse_results = op_metrics.iloc[
        op_metrics.groupby("iteration")["test_minority_rmse"].idxmin()
    ].loc[:, select_columns]

    # Write metrics to excel
    with pd.ExcelWriter(filename) as writer:
        op_metrics.to_excel(excel_writer=writer, sheet_name="op_metrics")

        total_mae_results.to_excel(excel_writer=writer, sheet_name="by_total_mae")
        majority_mae_results.to_excel(
            excel_writer=writer, sheet_name="by_majoritiy_mae"
        )
        minority_mae_results.to_excel(excel_writer=writer, sheet_name="by_minority_mae")
        total_mae_results.insert(0, "by", "total_mae")
        majority_mae_results.insert(0, "by", "majority_mae")
        minority_mae_results.insert(0, "by", "minority_mae")
        summary = pd.concat(
            [total_mae_results, majority_mae_results, minority_mae_results]
        )
        summary = summary.sort_values(by=["iteration", "by"], ascending=(True, True))
        summary.to_excel(excel_writer=writer, sheet_name="summary_mae")

        total_rmse_results.to_excel(excel_writer=writer, sheet_name="by_total_rmse")
        majority_rmse_results.to_excel(
            excel_writer=writer, sheet_name="by_majoritiy_rmse"
        )
        minority_rmse_results.to_excel(
            excel_writer=writer, sheet_name="by_minority_rmse"
        )
        total_rmse_results.insert(0, "by", "total_rmse")
        majority_rmse_results.insert(0, "by", "majority_rmse")
        minority_rmse_results.insert(0, "by", "minority_rmse")
        summary = pd.concat(
            [total_rmse_results, majority_rmse_results, minority_rmse_results]
        )
        summary = summary.sort_values(by=["iteration", "by"], ascending=(True, True))
        summary.to_excel(excel_writer=writer, sheet_name="summary_rmse")

    return op_metrics


def handle_key_value(op_result: np.array) -> dict:
    """Transform key and value arrays into a dict for parameters

    Args:
        op_result (np.array): the optimization result.

    Returns:
        dict: return parameters dict.

    """
    keys = op_result[0][-2]
    values = op_result[0][-1]
    d = defaultdict()
    for key, value in zip(keys, values):
        d[key] = value
    return d


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


def analyze_summarys(summarys: list) -> np.array:
    """Analysis summarys by the mean

    Args:
        summarys (list): a list including many metrics and iterations.

    Returns:
        np.array: return mean of summary metrics ``summarys_mean``.

    """
    summarys = pd.DataFrame(summarys)
    summarys_mean = []
    for _, col in summarys.iteritems():
        summarys_mean.append(col.mean())
    return summarys_mean


def save_metrics(metrics: list, csvfile: str = "summary.xlsx") -> None:
    """Save metrics into excel file

    Args:
        metrics (list): metrics for results.
        csvfile (str, optional): filename you want to save for metrics summary.
            Defaults to "summary.xlsx".

    """
    print_metrics = []
    it = iter(metrics)
    for i in range(3):
        print_metric = []
        print_metric.append(f"{next(it):6f} ± {next(it):6f}")
        print_metric.append(f"{next(it):6f} ± {next(it):6f}")
        print_metric.append(f"{next(it):6f} ± {next(it):6f}")
        print_metric.append(f"{next(it):6f}")
        print_metric.append(f"{next(it):6f}")
        print_metric.append(f"{next(it):6f}")
        print_metric.append(f"{next(it):6f}")
        # it = iter(range(10*i, 10*(i+1)))
        # print_metric.append(f"{metrics[next(it)]:6f} ± {metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f} ± {metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f} ± {metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[next(it)]:6f}")
        # print_metric.append(f"{metrics[0+10*i]:6f} ± {metrics[1+10*i]:6f}")
        # print_metric.append(f"%.6f ± %.6f" % (metrics[2 + 10 * i], metrics[3 + 10 * i]))
        # print_metric.append(f"%.6f ± %.6f" % (metrics[4 + 10 * i], metrics[5 + 10 * i]))
        # print_metric.append(f"%.6f" % metrics[6 + 10 * i])
        # print_metric.append(f"%.6f" % metrics[7 + 10 * i])
        # print_metric.append(f"%.6f" % metrics[8 + 10 * i])
        # print_metric.append(f"%.6f" % metrics[9 + 10 * i])
        print_metrics.append(print_metric)

    header = ["Training", "Validation", "Test"]
    print_metrics = pd.DataFrame(print_metrics).T
    with pd.ExcelWriter(csvfile) as writer:
        print_metrics.to_excel(
            excel_writer=writer, sheet_name="summary", header=header, index=None
        )


def index_to_name(data: pd.DataFrame, index: list) -> np.array:
    """Transform indexes into material names

    Args:
        data (pd.DataFrame): the original data.
        index (list): name index needed to be transformmed.

    Returns:
        np.array: return names array.

    """
    df = pd.DataFrame(data.iloc[:, 0])
    names = pd.DataFrame(df.iloc[index, :].reset_index(drop=True)).T
    names = names.to_numpy().flatten()
    return names


def save_result(
    results: pd.DataFrame,
    filename: str,
    iteration: int = 0,
    header: bool | list = None,
    index: bool | list = None,
) -> None:
    """Save result to excel file

    Args:
        results (pd.DataFrame): prediction result.
        filename (str): the filename you want to save result.
        iteration (int, optional): the number of iteration. Defaults to 0.
        header (bool or list, optional): whether to write the header or with the given
            header. Defaults to None.
        index (bool or list, optional): whether to write the index or with the given
            index. Defaults to None.

    """
    for i, result in enumerate(results):
        result = pd.DataFrame(result).T
        try:
            with pd.ExcelWriter(
                path=filename, mode="a", if_sheet_exists="replace"
            ) as writer:
                result.to_excel(
                    excel_writer=writer,
                    sheet_name=f"iteration-{(iteration*25+i)}",
                    header=header,
                    index=index,
                )
        except FileExistsError:
            with pd.ExcelWriter(path=filename, mode="w") as writer:
                result.to_excel(
                    excel_writer=writer,
                    sheet_name=f"iteration-{(iteration*25+i)}",
                    header=header,
                    index=index,
                )


def save_results(results: pd.DataFrame, filename: str, n_jobs: int = 25) -> None:
    """Save results to excel file or files

    Note:
        If iterations in ``results`` (the length of this list) are more than 25, results
        will be saved into serveral excel files in parallel automatically.

    Args:
        results (pd.DataFrame): prediction results.
        filename (str): the filename you want to save results.
        n_jobs (int, optional): the cores to be used, only available when iterations
            are more than 25.

    """
    iterations = len(results)
    if iterations > 25:
        filename = filename[:-5]
        _ = Parallel(n_jobs=n_jobs)(
            delayed(save_result)(results[i : i + 25], f"filename_{i}.xlsx", i)
            for i in tqdm(range(int(iterations / 25)))
        )
    else:
        save_result(results, filename)
    print(f"Save all results into excel files for `{filename}` successfully.")


def load_result(filename: str, sheet_name: str) -> np.array:
    """Load result from excel with specified sheet

    Args:
        filename (str): the excel file name.
        sheet_name (str): the specified sheet name.

    Raises:
        FileNotFoundError: If the specified file or sheet is not found.

    Returns:
        np.array: return ``result`` for one iteration from excel file.

    """
    try:
        data = pd.read_excel(filename, sheet_name)
        data = data.T
        result = []
        for _, row in data.iterrows():
            result.append(row.dropna(axis=0).values)
        return result
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} not found.")


def load_results(filename: str, parallel: bool = False) -> np.array:
    """Load results from excel with all sheets

    Args:
        filename (str): the excel file name (with path).
        parallel (bool, optional): whether parallel or not. Defaults to False.

    Raises:
        FileNotFoundError: If the specified file is not found.

    Returns:
        np.array: return ``results`` from excel file.

    """
    try:
        data_results = pd.ExcelFile(filename)
        sheet_names = data_results.sheet_names
        results = []
        start_at = time.time()
        if parallel:
            results = Parallel(n_jobs=10)(
                delayed(load_result)(filename, sheet_name) for sheet_name in sheet_names
            )
        else:
            for sheet_name in sheet_names:
                result = load_result(filename, sheet_name)
                results.append(result)
        end_at = time.time()
        if parallel:
            print(f"Loading {filename} used {(end_at - start_at)}(s) " f"with parallel")
        else:
            print(
                f"Loading {filename} used {(end_at - start_at)}(s) " f"without parallel"
            )
        return results
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} not found.")


def load_summarys(metrics_filename: str) -> np.array:
    """Load metrics summary from excel file

    Args:
        metrics_filename (str): the excel file name.

    Raises:
        FileNotFoundError: If the specified file is not found.

    Returns:
        np.array: return ``op_summarys`` for each C.

    """
    try:
        data_metrics = pd.read_excel(metrics_filename, "op_metrics", index_col=0)
        C_list = pd.unique(data_metrics["C"])
        data_metrics_mean = data_metrics.groupby("C").mean().iloc[:, 1:]
        op_summarys = []

        for C in C_list:
            op_summary = []
            op_summary.append([C])
            op_summary.append(data_metrics_mean.loc[C].values)
            op_summarys.append(op_summary)

        return op_summarys
    except FileNotFoundError:
        raise FileNotFoundError(f"{metrics_filename} is not found.")


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
        try:
            with pd.ExcelWriter(
                path=filename, mode="a", if_sheet_exists="replace"
            ) as writer:
                result.to_excel(
                    excel_writer=writer,
                    sheet_name=f"iteration-{(iteration*25+i)}",
                    index=None,
                )
        except FileExistsError:
            with pd.ExcelWriter(path=filename, mode="w") as writer:
                result.to_excel(
                    excel_writer=writer,
                    sheet_name=f"iteration-{(iteration*25+i)}",
                    index=None,
                )


def analyze_costs(
    siml_results: pd.DataFrame,
    results: pd.DataFrame,
    filename: str = "costs.xlsx",
) -> np.array:
    """Analyze siml method results
    
    Note:
        It is only available for ``siml`` learning method.

    Args:
        siml_results (pd.DataFrame): siml method results.
        results (pd.DataFrame): results by majority training set and only selected
            minority instance.
        filename (str, optional): filename you want to save analysis results.
            Defaults to "costs.xlsx".

    Raises:
        NameError: If the lengths of two results are not the same.

    Returns:
        np.array: return analysis results.

    """
    iteration1 = len(siml_results)
    iteration2 = len(siml_results)
    if iteration1 == iteration2:
        iteration = iteration1
        l12_summary = []
        for i in range(iteration):
            l1 = pd.DataFrame(
                {
                    "Bandgap": siml_results[i][10],
                    "Prediction": siml_results[i][11],
                    "Material": siml_results[i][9],
                    "Weights": siml_results[i][19],
                }
            )
            l2 = pd.DataFrame(
                {
                    "Bandgap": results[i][1],
                    "Material": results[i][2],
                    "Origin": results[i][3],
                }
            )
            l12 = pd.merge(l1, l2, how="left", on=["Bandgap", "Material"])
            l12["Difference"] = l12["Weights"] - l12["Origin"]
            l12.insert(0, "No", range(1, 1 + len(l12)))
            l12_summary.append(l12)

        save_analyzed_result(l12_summary, filename)
        return l12_summary
    else:
        raise NameError(f"Two results are not match! Please check them firstly!")
