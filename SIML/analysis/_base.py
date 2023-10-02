"""Basic functions for analysis"""

from collections import defaultdict

import numpy as np
import pandas as pd


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


def metrics2msg(metrics: list, method: str = "SIML") -> str:
    """Format metrics to readable strings

    Args:
        metrics (list): metric values.
        method (str, optional): the method name, could be anything. Defaults to "SIML".

    Returns:
        str: return readable strings for metrics.
    """
    names = ["Metrics", "Training", "Validation", "Test"]
    msg = (
        f"**************************************\n"
        f"The results of {method}:\n\n"
        f"{names[0]:<11}{names[1]:<20}{names[2]:<20}{names[3]}\n"
        f"MAE        {metrics[0]:.3f} ± {metrics[1]:<12.3f}"
        f"{metrics[12]:.3f} ± {metrics[13]:<12.3f}"
        f"{metrics[24]:.3f} ± {metrics[25]:.3f}\n"
        f"Maj MAE    {metrics[2]:.3f} ± {metrics[3]:<12.3f}"
        f"{metrics[14]:.3f} ± {metrics[15]:<12.3f}"
        f"{metrics[26]:.3f} ± {metrics[27]:.3f}\n"
        f"Min MAE    {metrics[4]:.3f} ± {metrics[5]:<12.3f}"
        f"{metrics[16]:.3f} ± {metrics[17]:<12.3f}"
        f"{metrics[28]:.3f} ± {metrics[29]:.3f}\n"
        f"Variance   {metrics[6]:<20.3f}{metrics[18]:<20.3f}{metrics[30]:.3f}\n"
        f"RMSE       {metrics[7]:<20.3f}{metrics[19]:<20.3f}{metrics[31]:.3f}\n"
        f"Maj RMSE   {metrics[8]:<20.3f}{metrics[20]:<20.3f}{metrics[32]:.3f}\n"
        f"Min RMSE   {metrics[9]:<20.3f}{metrics[21]:<20.3f}{metrics[33]:.3f}\n"
        f"MAPE       {metrics[10]:<20.3f}{metrics[22]:<20.3f}{metrics[34]:.3f}\n"
        f"R2         {metrics[11]:<20.3f}{metrics[23]:<20.3f}{metrics[35]:.3f}\n"
        f"**************************************"
    )
    return msg


def metrics2array(metrics: list) -> list:
    results = []
    it = iter(metrics)
    for _ in range(3):
        result = []
        result.append(f"{next(it)} ± {next(it)}")
        result.append(f"{next(it)} ± {next(it)}")
        result.append(f"{next(it)} ± {next(it)}")
        result.append(f"{next(it)}")
        result.append(f"{next(it)}")
        result.append(f"{next(it)}")
        result.append(f"{next(it)}")
        result.append(f"{next(it)}")
        result.append(f"{next(it)}")
        results.append(result)
    return results
