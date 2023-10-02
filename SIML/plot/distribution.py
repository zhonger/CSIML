"""Distribution figures for dataset"""
import math

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ._base import (
    cal_distribution,
    cal_distribution_circle,
    cal_distribution_elements,
    cal_distribution_period,
    make_autopct,
    plot_post,
)
from ._dataclass import Cells, ElementC


def plot_histogram(data: pd.DataFrame, filename: str = None, **kws) -> None:
    """Plot histogram for dataset property distribution

    Args:
        data (pd.DataFrame): the dataset you want to evaluate.
        filename (str, optional): the filename. If it's not specified, it will only be
            shown in interactive way. If it's specified, it will be saved to the file.
            Defaults to None.
        num_bins (int, optional): the num of bins. Defaults to 80.
        ranges (list, optional): the range of property values. Defaults to [-2, 22].
        dpi (int, optional): the dpi value for figures. Defaults to 100.
        xlabel (str, optional): the x axis label name. Defaults to "Experimental
            bandgaps (eV)".
        ylabel (str, optional): the y axis label name. Defaults to "Probability
            density".
        color (str, optional): the color name for filled area. Defaults to "orange".
        facecolor (str, optional): the color name for histogram. Defaults to "bisque".
        edgecolor (str, optional): the color name for histogram edge. Defaults to
            "black".
    """
    x = data["Experimental"].values
    num_bins = kws.get("num_bins", 80)
    ranges = kws.get("ranges", [-2, 22])
    dpi = kws.get("dpi", 100)
    xlabel = kws.get("xlabel", "Experimental bandgaps (eV)")
    ylabel = kws.get("ylabel", "Probability density")
    color = kws.get("color", "orange")
    facecolor = kws.get("facecolor", "bisque")
    edgecolor = kws.get("edgecolor", "black")

    fig, ax = plt.subplots(dpi=dpi)
    ax.hist(x, num_bins, ranges, density=True, facecolor=facecolor, edgecolor=edgecolor)
    sns.kdeplot(x, fill=True, color=color, alpha=0.3)
    fontdict = {"weight": "bold"}
    ax.set_xlabel(xlabel, fontdict=fontdict)
    ax.set_ylabel(ylabel, fontdict=fontdict)
    ax.set_xlim(ranges)
    fig.tight_layout()
    plot_post(filename, dpi)


def plot_bar(
    data: pd.DataFrame,
    level1: float = 3.0,
    level2: float = 5.0,
    filename: str = None,
    **kws,
) -> None:
    """Plot the distribution according to the number of elements

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.0.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.0.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
        unit (str, optional): the unit of property value. Defaults to "eV".
        dpi (int, optional): the dpi value for the figure. Defaults to 100.
        colors (list, optional): the color list including 3 colors. Defaults to
            ["bisque", "orange", "royalblue"].
        y_max (float, optional): the max value for rectangle height. Defaults from the
            calculation of statistical values.
        width (float, optional): the width of each rectangle. Defaults to 0.2.
    """
    unit = kws.get("unit", "eV")
    dpi = kws.get("dpi", 100)
    colors = kws.get("colors", ["bisque", "orange", "royalblue"])
    statss, y_max = cal_distribution_elements(data, level1, level2, unit)
    y_max = kws.get("y_max", y_max)
    width = kws.get("width", 0.2)
    species = ("1", "2", "3", "4")
    x = np.arange(len(species))

    _, ax = plt.subplots(dpi=dpi)
    plot_bar_rect(ax, colors, statss, x, width)
    ax.set_xlabel("# Elements", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xticks(x + width, species)
    ax.legend(loc="upper left", ncols=3, frameon=False)
    ax.set_ylim(0, y_max)
    plot_post(filename, dpi)


def plot_bar_rect(
    ax: plt.Axes, colors: list, statss: dict, x: np.array, width: float
) -> None:
    """Plot rectangles for element bars.

    Args:
        ax (plt.Axes): the axes object.
        colors (list): the color list including 3 colors.
        statss (dict): the statistical values.
        x (np.array): the label locations.
        width (float): the width for each rectangle.
    """
    index = 0
    for label, count in statss.items():
        offset = width * index
        rect = ax.bar(x + offset, count, width, label=label, color=colors[index % 3])
        ax.bar_label(rect, padding=3)
        index += 1


def plot_period(
    data: pd.DataFrame,
    level1: float = 3,
    level2: float = 5,
    filename: str = None,
    **kws,
) -> None:
    """Plot the distribution according to periodic table

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
        dpi (int, optional): the dpi value of the figure. Defaults to 100.
        xymax (tuple, optional): the x and y max value. Defaults to (20, 11).
        cell_length (float, optional): the length of each cell. Defaults to 1.
        cell_gap (float, optional): the gap between cells. Defaults to 0.1.
        cell_edge_width (float, optional): the edge width of cells. Defaults to 0.5.
    """
    dpi = kws.get("dpi", 100)
    xymax = kws.get("xymax", (20, 11))
    cell_length = kws.get("cell_length", 1)
    cell_gap = kws.get("cell_gap", 0.1)
    cell_edge_width = kws.get("cell_edge_width", 0.5)

    _, results_summary = cal_distribution(data, level1, level2)
    elements = cal_distribution_period(results_summary)

    plt.figure(figsize=(10, 5), dpi=dpi)
    my_cmap = mpl.cm.get_cmap(kws.get("cmap", "RdYlGn"))
    norm = mpl.colors.Normalize(1, 111)
    cells = Cells(cell_length, cell_edge_width, cell_gap, my_cmap, norm)
    my_cmap.set_under("None")
    plt.colorbar(mpl.cm.ScalarMappable(norm, my_cmap), drawedges=False)
    plot_period_elements(elements, cells, xymax[1])
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.ylim(0, xymax[1])
    plt.xlim(0, xymax[0])
    plot_post(filename, dpi)


def plot_period_elements(elements: list, cells: Cells, y_max: float) -> None:
    """Plot all periodic elements

    Args:
        elements (list): elements information.
        cells (Cells): settings for element cells.
        y_max (float): the maximum of y (position) in the whole figure.
    """
    for e in elements:
        if e.group is None:
            continue
        x = (cells.cell_length + cells.cell_gap) * (e.group - 1)
        y = y_max - ((cells.cell_length + cells.cell_gap) * e.period)
        if e.period >= 8:
            y -= cells.cell_length * 0.5
        plot_period_element(x, y, e, cells)


def plot_period_element(x: float, y: float, e: ElementC, cells: Cells) -> None:
    """Plot one periodic element

    Args:
        x (float): the x axis position.
        y (float): the y axis position.
        e (ElementC): the element information.
        cells (Cells): settings for element cells.
    """
    fontdict1 = {"size": 6, "color": "black"}
    fontdict2 = {"size": 9, "color": "black", "weight": "bold"}
    if e.number:
        fill_color = cells.my_cmap(cells.norm(e.count))
        rect = mpl.patches.Rectangle(
            xy=(x, y),
            width=cells.cell_length,
            height=cells.cell_length,
            linewidth=cells.cell_edge_width,
            edgecolor="k",
            facecolor=fill_color,
        )
        plt.gca().add_patch(rect)
    plt.text(x + 0.04, y + 0.8, e.number, va="center", ha="left", fontdict=fontdict1)
    plt.text(x + 0.5, y + 0.5, e.symbol, va="center", ha="center", fontdict=fontdict2)
    plt.text(x + 0.5, y + 0.12, e.count, va="center", ha="center", fontdict=fontdict1)


def plot_circle(
    data: pd.DataFrame,
    level1: float = 3,
    level2: float = 5,
    filename: str = None,
    **kws,
) -> None:
    """Plot the distribution according to existed elements

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
    """
    dpi = kws.get("dpi", 100)
    unit = kws.get("unit", "eV")
    labels = [f"0~{level1} {unit}", f"{level1}~{level2} {unit}", f">{level2} {unit}"]
    colors = kws.get("colors", ["bisque", "orange", "royalblue"])
    textprops = kws.get("textprops", {"weight": "bold", "size": 14})

    total, results = cal_distribution_circle(data, level1, level2)
    fig, axs = plt.subplots(math.ceil(total / 12), 12, figsize=(20, 10))
    plot_circle_element(axs, results, colors, textprops)
    fig.tight_layout()
    fig.legend(labels=labels, loc="lower right", prop=textprops)
    plot_post(filename, dpi)


def plot_circle_element(
    axs: plt.Axes, results: pd.DataFrame, colors: list, textprops: dict
) -> None:
    """Plot circle (pie) for all existed elements

    Args:
        axs (plt.Axes): the Axes obejct.
        results (pd.DataFrame): statistical results for each element.
        colors (list): the color list for different number of elements.
        textprops (dict): text style settings.
    """
    i = 0
    j = 0
    for key in results.keys():
        axs[i, j].pie(
            results[key],
            autopct=make_autopct(results[key]),
            colors=colors,
            textprops=textprops,
        )
        axs[i, j].set_title(key, fontsize="16", fontweight="bold")
        if j < 11:
            j += 1
        else:
            i += 1
            j = 0
    for k in range(j, 12):
        axs[i, k].axis("off")
