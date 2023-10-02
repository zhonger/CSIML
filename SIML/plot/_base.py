"""Base classes or functions for plot module"""
import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matminer.featurizers.conversions import StrToComposition
from matplotlib import pyplot as plt
from pymatgen.core.composition import Composition

from ._dataclass import ElementC
from .periodic_table import PT


def get_elements_length(composition: Composition) -> int:
    """Calculate the number of elements

    Args:
        composition (Composition): the composition object for specified material.

    Returns:
        int: the number of elements in specified material.
    """
    return len(composition.elements)


def get_level(value: float, level1: float = 3.0, level2: float = 5.0) -> int:
    """Calculate the level of the property value

    Args:
        value (float): the property value, for example "bandgap" value.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.0.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.0.

    Returns:
        int: the level of the property value.
    """
    if value < level1:
        return 1
    if value < level2:
        return 2
    return 3


def count_errors(errors: np.array, t1: float = 0.5, t2: float = 1) -> list:
    """Count nums according to the levels of error

    Args:
        errors (np.array): the errors between experimental and predicted values.
        t1 (float): the threshold of acceptable error.
        t2 (float): the threshold of barely acceptable error.

    Returns:
        list: the metrics for errors.
    """
    num = []
    total = errors.shape[0]
    errors = pd.DataFrame(abs(errors))

    c1 = errors[errors <= t1].count().iloc[0]
    c2 = errors[t1 < errors][errors <= t2].count().iloc[0]
    c3 = errors[errors > t2].count().iloc[0]
    num.append(c1)
    num.append(c2)
    num.append(c3)
    num.append(round(c1 / total * 100, 2))
    num.append(round(c2 / total * 100, 2))
    num.append(round(c3 / total * 100, 2))
    return num
    # emetric = EMetric(
    #     c1,
    #     c2,
    #     c3,
    #     round(c1 / total * 100, 2),
    #     round(c2 / total * 100, 2),
    #     round(c3 / total * 100, 2),
    # )

    # return emetric


def cal_distribution_elements(
    data: pd.DataFrame, level1: float = 3.0, level2: float = 5.0, unit: str = "eV"
) -> Tuple[dict, float]:
    """Calculate the distribution of numbers of elements

    Args:
        data (pd.DataFrame): the data with materials name and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3 (for
            bandgap).
        level2 (float, optional): the threshold value of level 2. Dafaults to 5 (for
            bandgap).
        unit (str, optional): the unit name of the property. Defaults to "eV".

    Returns:
        Tuple[dict, float]: the statistical values and maximum value.
    """
    df = StrToComposition().featurize_dataframe(
        data[["Material", "Experimental"]], "Material"
    )
    df["elements"] = df["composition"].map(get_elements_length)
    df["level"] = df["Experimental"].apply(lambda x: get_level(x, level1, level2))
    counts = df.iloc[:, 3:].groupby(["elements", "level"]).value_counts()
    counts_max = counts.max()
    if counts_max < 500:
        y_max = math.ceil(counts_max / 100) * 100
    else:
        y_max = math.ceil(counts_max / 100) * 100 + 500
    counts = counts.to_dict()
    stats = np.zeros([3, 4], dtype=np.int64)
    for e, l in counts:
        stats[l - 1][e - 1] = counts[(e, l)]
    statss = {
        f"0~{level1} {unit}": tuple(stats[0]),
        f"{level1}~{level2} {unit}": tuple(stats[1]),
        f">{level2} {unit}": tuple(stats[2]),
    }
    return statss, y_max


def cal_distribution(
    data: pd.DataFrame, level1: float = 3, level2: float = 5
) -> Tuple[list, list]:
    """Calculate the distribution of the property

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float): the threshold value of level 1. Defaults to 3 (for bandgap).
        level2 (float): the threshold value of level 2. Defaults to 5 (for bandgap).

    Returns:
        Tuple[list, list]: return the distribution according to existed elements and
        periodic table.
    """
    results = {}
    results_summary = {}
    df = StrToComposition().featurize_dataframe(
        data[["Material", "Experimental"]], "Material"
    )
    PThelper = PT()
    for i in range(0, df.shape[0]):
        cmp = df.iloc[i]["composition"]
        bandgap = df.iloc[i]["Experimental"]
        element_num = len(cmp.elements)
        for j in range(0, element_num):
            element = str(cmp.elements[j])
            if element not in results:
                results[element] = [0, 0, 0, PThelper.get_element_number(element)]
                results_summary[element] = 0
            if bandgap <= level1:
                results[element][0] += 1
            elif bandgap <= level2:
                results[element][1] += 1
            else:
                results[element][2] += 1
            results_summary[element] += 1

    distribution = pd.DataFrame(results)
    distribution_sum = results_summary

    return distribution, distribution_sum


def cal_distribution_period(results_summary):
    elements = []
    PThelper = PT()
    for i in range(1, 119):
        ele = PThelper.get_element(number=i)
        ele_group, ele_period = ele.group, ele.period
        if 57 <= i <= 71:
            ele_group, ele_period = i - 57 + 3, 8
        if 89 <= i <= 103:
            ele_group, ele_period = i - 89 + 3, 9
        ele_count = results_summary.setdefault(ele.symbol, 0)
        elements.append(ElementC(i, ele.symbol, ele_group, ele_period, ele_count))
    elements.append(ElementC(None, "LA", 3, 6, None))
    elements.append(ElementC(None, "AC", 3, 7, None))
    elements.append(ElementC(None, "LA", 2, 8, None))
    elements.append(ElementC(None, "AC", 2, 9, None))
    return elements


def plot_post(filename: str, dpi: int):
    if filename:
        plt.savefig(filename, dpi=dpi)
    else:
        plt.show()


def plotly_post(fig: go.Figure, filename: str):
    if filename:
        fig.write_html(f"{filename}.html")
    else:
        fig.show()


def cal_distribution_circle(
    data: pd.DataFrame, level1: float, level2: float
) -> Tuple[int, pd.DataFrame]:
    distribution, _ = cal_distribution(data, level1, level2)
    dt = distribution.T
    dt.columns = ["col1", "col2", "col3", "col4"]
    dt.sort_values(by=["col4"], inplace=True)
    results = dt.T.iloc[:3]
    return len(dt), results


def make_autopct(values: list) -> str:
    """Make label for pie chart automatically

    Args:
        values (list): level distribution for specified element.

    Return:
        str: the label.
    """

    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return "" if val == 0 else f"{val:d}"

    return my_autopct


def get_names(method: str) -> list:
    names1 = [
        "Train (Majority)",
        "Validation (Majority)",
        "Test (Majority)",
        "Test (Minority)",
    ]
    names2 = [
        "Train (Majority)",
        "Validation (Majority)",
        "Test (Majority)",
        "Train (Minority)",
        "Validation (Minority)",
        "Test (Minority)",
    ]
    names = names1 if method == "basis1" else names2
    return names
