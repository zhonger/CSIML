"""Interface with files for analysis"""

import pickle
import time
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def write_excel(result: pd.DataFrame, path: str, mode: str, **kws):
    """Write pandas dataframe to excel file

    Args:
        result (pd.DataFrame): the result data neede to be saved.
        path (str): the path to save including filename.
        mode (str): writing mode, like "a" and "w".
        if_sheet_exists (bool | str, optional): how to handle when specified sheet
            exists. It could be "replace" or None. Defaults to None.
        sheet_name (str, optional): the name of sheet. Defaults to "Sheet1".
        header (bool or list[str], optional): the header of sheet. Defaults to None.
        index (bool or list, optional): the index of sheet. Defaults to None.
    """
    if_sheet_exists = kws.get("if_sheet_exists", None)
    sheet_name = kws.get("sheet_name", "Sheet1")
    header = kws.get("header", None)
    index = kws.get("index", None)
    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        path, mode=mode, if_sheet_exists=if_sheet_exists
    ) as writer:
        result.to_excel(
            excel_writer=writer,
            sheet_name=sheet_name,
            header=header,
            index=index,
        )


def load_pkl(filename: str) -> list:
    """Load results from pkl binary file

    Args:
        filename (str): The filename used including the path.

    Raises:
        FileNotFoundError: if the specified file cannot be found.

    Returns:
        list: return the results list object.
    """
    try:
        with open(filename, "rb") as file:
            results = pickle.load(file)
        return results
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{filename} not found") from e


def write_pkl(results: Any, filename: str) -> None:
    """Write results to pkl binary file

    Args:
        results (list | np.array | pd.DataFrame): python object.
        filename (str): the filename you want to save into.
    """
    with open(filename, "wb") as file:
        pickle.dump(results, file)


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
        sheet_name = f"iteration-{(iteration*25+i)}"
        kws = {"sheet_name": sheet_name, "header": header, "index": index}
        try:
            write_excel(result, filename, "a", if_sheet_exists="replace", **kws)
        except FileExistsError:
            write_excel(result, filename, "w", **kws)


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
            delayed(save_result)(results[i : i + 25], f"{filename}_{i}.xlsx", i)
            for i in tqdm(range(int(iterations / 25)))
        )
    else:
        save_result(results, filename)
    print(f"Save all results into excel files for `{filename}` successfully.")


def load_result(filename: str, sheet_name: str) -> list:
    """Load result from excel with specified sheet

    Args:
        filename (str): the excel file name.
        sheet_name (str): the specified sheet name.

    Raises:
        FileNotFoundError: If the specified file or sheet is not found.

    Returns:
        list: return ``result`` for one iteration from excel file.
    """
    try:
        data = pd.read_excel(filename, sheet_name)
        data = data.T
        result = []
        for _, row in data.iterrows():
            result.append(row.dropna(axis=0).values)
        return result
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{filename} not found.") from e


def load_results(filename: str, n_jobs: int = 1) -> list:
    """Load results from excel with all sheets

    Args:
        filename (str): the excel file name (with path).
        n_jobs (int, optional): the number of cores. Defaults to 1.

    Raises:
        FileNotFoundError: If the specified file is not found.

    Returns:
        list: return ``results`` from excel file.
    """
    try:
        data_results = pd.ExcelFile(filename)
        sheet_names = data_results.sheet_names
        results = []
        start_at = time.time()
        results = Parallel(n_jobs=n_jobs)(
            delayed(load_result)(filename, sheet_name) for sheet_name in sheet_names
        )
        end_at = time.time()
        print(f"Loading {filename} used {(end_at - start_at)}(s)")
        return results
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{filename} not found.") from e


def load_summarys(filename: str, sheet_name: str = "op_metrics") -> list:
    """Load metrics summary from excel file

    Args:
        filename (str): the excel file name.

    Raises:
        FileNotFoundError: If the specified file is not found.

    Returns:
        list: return ``op_summarys`` for each C.
    """
    try:
        data_metrics = pd.read_excel(filename, sheet_name, index_col=0)
        C_list = pd.unique(data_metrics["C"])
        data_metrics_mean = data_metrics.groupby("C").mean().iloc[:, 1:]
        op_summarys = []
        for C in C_list:
            op_summary = []
            op_summary.append([C])
            op_summary.append(data_metrics_mean.loc[C].values)
            op_summarys.append(op_summary)
        return op_summarys
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{filename} is not found.") from e
