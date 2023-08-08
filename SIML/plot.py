import math
from dataclasses import dataclass
from typing import Tuple

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mendeleev
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matminer.featurizers.conversions import StrToComposition
from matplotlib import rcParams


@dataclass
class EMetric:
    """A dataclass for error metrics

    Args:
        c1 (int): the number of acceptable error.
        c2 (int): the number of barely acceptable error.
        c3 (int): the number of not acceptable error.
        cp1 (float): the proportion of acceptable error in all errors.
        cp2 (float): the proportion of barely acceptable error in all errors.
        cp3 (float): the proportion of not acceptable error in all errors.

    """

    c1: int
    c2: int
    c3: int
    cp1: float
    cp2: float
    cp3: float


def Count(errors: np.array, t1: float = 0.5, t2: float = 1) -> list:
    """Count nums according to the levels of error

    Args:
        errors (np.array): the errors between experimental and predicted values.
        t1 (float): the threshold of acceptable error.
        t2 (float): the threshold of barely acceptable error.

    Returns:
        list: the metrics for errors.

    """
    num = []
    sum = errors.shape[0]
    errors = pd.DataFrame(abs(errors))

    c1 = errors[errors <= t1].count().iloc[0]
    c2 = errors[t1 < errors][errors <= t2].count().iloc[0]
    c3 = errors[errors > t2].count().iloc[0]
    num.append(c1)
    num.append(c2)
    num.append(c3)
    num.append(round(c1 / sum * 100, 2))
    num.append(round(c2 / sum * 100, 2))
    num.append(round(c3 / sum * 100, 2))
    return num
    # emetric = EMetric(
    #     c1,
    #     c2,
    #     c3,
    #     round(c1 / sum * 100, 2),
    #     round(c2 / sum * 100, 2),
    #     round(c3 / sum * 100, 2),
    # )

    # return emetric


def auto_label(rects) -> None:
    """Make labels for data distribution bars

    Args:
        rects (matplotlib.pyplot): the matplotlib.pyplot object to be used.

    """
    for rect in rects:
        height = rect.get_height()
        plt.annotate(
            "{}".format(height),  # put the detail data
            xy=(
                rect.get_x() + rect.get_width() / 2,
                height,
            ),  # get the center location.xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="baseline",
        )


def PlotBar(
    data: pd.DataFrame,
    level1: float = 3.0,
    level2: float = 5.0,
    filename: str = None,
) -> None:
    """Plot the distribution according to the number of elements

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.

    """
    nums = [0, 0, 0, 0]
    stats = np.zeros([4, 3], dtype=np.int64)

    df = StrToComposition().featurize_dataframe(
        data[["Material", "Experimental"]], "Material"
    )

    for i in range(0, len(df)):
        length = len(df.iloc[i]["composition"].elements)
        df.loc[i, "elements"] = length
        nums[length - 1] += 1
        if df.iloc[i]["Experimental"] < level1:
            stats[length - 1][0] += 1
        elif df.iloc[i]["Experimental"] < level2:
            stats[length - 1][1] += 1
        else:
            stats[length - 1][2] += 1

    stats = stats.T

    width = 0.2
    sname = ["1", "2", "3", "4"]
    index = np.arange(len(sname))
    labels = [
        f"0~{level1} eV",
        f"{level1}~{level2} eV",
        f">{level2} eV",
    ]
    colors = ["bisque", "orange", "royalblue"]

    r1 = plt.bar(index - width, stats[0], width, color=colors[0], label=labels[0])
    r2 = plt.bar(index, stats[1], width, color=colors[1], label=labels[1])
    r3 = plt.bar(index + width, stats[2], width, color=colors[2], label=labels[2])
    auto_label(r1)
    auto_label(r2)
    auto_label(r3)

    plt.xticks(index, labels=sname)
    plt.xlabel("# Elements", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()


def CalDistribution(
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
    results = dict()
    results_summary = dict()
    df = StrToComposition().featurize_dataframe(
        data[["Material", "Experimental"]], "Material"
    )

    for i in range(0, df.shape[0]):
        cmp = df.iloc[i]["composition"]
        bandgap = df.iloc[i]["Experimental"]
        element_num = len(cmp.elements)
        for j in range(0, element_num):
            element = str(cmp.elements[j])
            if element not in results:
                med = mendeleev.element(element)
                results[element] = [0, 0, 0, med.atomic_number]
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


def PlotPeriod(
    data: pd.DataFrame, level1: float = 3, level2: float = 5, filename: str = None
) -> None:
    """Plot the distribution according to periodic table

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.
        filename (str, optional): the filename you want to save the figure.

    """
    cell_length = 1
    cell_gap = 0.1
    cell_edge_width = 0.5

    _, results_summary = CalDistribution(data, level1, level2)
    elements = []

    for i in range(1, 119):
        ele = mendeleev.element(i)
        ele_group, ele_period = ele.group_id, ele.period

        if 57 <= i <= 71:
            ele_group = i - 57 + 3
            ele_period = 8
        if 89 <= i <= 103:
            ele_group = i - 89 + 3
            ele_period = 9

        elements.append(
            [
                i,
                ele.symbol,
                ele_group,
                ele_period,
                results_summary.setdefault(ele.symbol, 0),
            ]
        )

    elements.append([None, "LA", 3, 6, None])
    elements.append([None, "AC", 3, 7, None])
    elements.append([None, "LA", 2, 8, None])
    elements.append([None, "AC", 2, 9, None])

    plt.figure(figsize=(10, 5))
    xy_length = (20, 11)

    my_cmap = cm.get_cmap("RdYlGn")
    norm = mpl.colors.Normalize(1, 111)
    my_cmap.set_under("None")
    cmmapable = cm.ScalarMappable(norm, my_cmap)
    plt.colorbar(cmmapable, drawedges=False)

    for e in elements:
        ele_number, ele_symbol, ele_group, ele_period, ele_count = e

        if ele_group is None:
            continue

        x = (cell_length + cell_gap) * (ele_group - 1)
        y = xy_length[1] - ((cell_length + cell_gap) * ele_period)

        if ele_period >= 8:
            y -= cell_length * 0.5

        if ele_number:
            fill_color = my_cmap(norm(ele_count))
            rect = patches.Rectangle(
                xy=(x, y),
                width=cell_length,
                height=cell_length,
                linewidth=cell_edge_width,
                edgecolor="k",
                facecolor=fill_color,
            )
            plt.gca().add_patch(rect)

        plt.text(
            x + 0.04,
            y + 0.8,
            ele_number,
            va="center",
            ha="left",
            fontdict={"size": 6, "color": "black"},
        )
        plt.text(
            x + 0.5,
            y + 0.5,
            ele_symbol,
            va="center",
            ha="center",
            fontdict={
                "size": 9,
                "color": "black",
                "weight": "bold",
            },
        )
        plt.text(
            x + 0.5,
            y + 0.12,
            ele_count,
            va="center",
            ha="center",
            fontdict={"size": 6, "color": "black"},
        )

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.ylim(0, xy_length[1])
    plt.xlim(0, xy_length[0])

    if filename:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()


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
        # return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        if val == 0:
            return ""
        else:
            return "{v:d}".format(v=val)

    return my_autopct


def PlotCirle(
    data: pd.DataFrame, level1: float = 3, level2: float = 5, filename: str = None
) -> None:
    """Plot the distribution according to existed elements

    Args:
        data (pd.DataFrame): the data with features and property.
        level1 (float, optional): the threshold value of level 1. Defaults to 3.
        level2 (float, optional): the threshold value of level 2. Defaults to 5.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.

    """
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "bold"
    rcParams["font.size"] = "12"

    distribution, _ = CalDistribution(data, level1, level2)
    dt = distribution.T
    dt.columns = ["col1", "col2", "col3", "col4"]
    dt.sort_values(by=["col4"], inplace=True)
    results = dt.T.iloc[:3]

    labels = [
        "0~" + str(level1) + " eV",
        str(level1) + "~" + str(level2) + " eV",
        ">" + str(level2) + " eV",
    ]
    colors = ["bisque", "orange", "royalblue"]

    total = len(dt)
    fig, axs = plt.subplots(math.ceil(total / 12), 12, figsize=(20, 10))

    keys = results.keys()
    i = 0
    j = 0

    for key in keys:
        axs[i, j].pie(results[key], autopct=make_autopct(results[key]), colors=colors)
        axs[i, j].set_title(key, fontsize="16", fontweight="bold")
        if j < 11:
            j += 1
        else:
            i += 1
            j = 0

    for k in range(j, 12):
        axs[i, k].axis("off")

    fig.tight_layout()
    fig.legend(labels=labels, loc="lower right")

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=600)


def PlotPrediction(result: np.array, method: str, filename: str = None) -> None:
    """Plot prediction-experimental values

    Args:
        result (np.array): prediction result.
        method (str): the method name.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.

    """
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
    symbols = ["circle", "square", "diamond", "star"]
    colors = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 13)",
        "rgb(44, 160, 44)",
        "rgb(214, 38, 40)",
        "rgb(148, 103, 189)",
        "rgb(140, 86, 75)",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-3, 16] + [16, -3],
            y=[-2, 17] + [15, -4],
            fill="toself",
            fillcolor="rgba(226, 201, 212, 1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="1 eV",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-3, 16] + [16, -3],
            y=[-2.5, 16.5] + [15.5, -3.5],
            fill="toself",
            fillcolor="rgba(204, 219, 214, 1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="0.5 eV",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-3, 16],
            y=[-3, 16],
            marker_color="red",
            line_dash="dash",
            name="Best line",
        )
    )

    length = 6 if int(len(result) / 3) > 6 else int(len(result) / 3)

    match method:
        case "basis1":
            names = names1
        case _:
            names = names2

    title = filename

    for i in range(length):
        fig.add_trace(
            go.Scatter(
                x=result[1 + 3 * i],
                y=result[2 + 3 * i],
                hovertext=result[3 * i],
                mode="markers",
                name=names[i],
                marker=dict(
                    size=14, color=colors[i], line=dict(width=2, color="black")
                ),
            )
        )

    # fig.add_trace(
    #     go.Scatter(
    #         x=[-3, 16],
    #         y=[-3, 16],
    #         error_y=[2, 2],
    #         error_y_mode="band",
    #         marker_color="red",
    #         line_dash="dash",
    #         name="Best line",
    #     )
    # )
    fig.add_vline(x=5, line=dict(width=3, dash="dash", color="red"))
    fig.add_vrect(
        x0=0,
        x1=5,
        annotation_text="Majority",
        annotation_position="top",
        annotation_font_size=24,
        fillcolor="skyblue",
        opacity=0.5,
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=5,
        x1=15,
        annotation_text="Minority",
        annotation_position="top",
        annotation_font_size=24,
        fillcolor="lightgray",
        opacity=0.5,
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        title="<b>" + title + "<b>",
        titlefont=dict(size=24),
        xaxis_title="Experimental (eV)",
        yaxis_title="Prediction (eV)",
        width=1300,
        height=1200,
        paper_bgcolor="white",
        showlegend=True,
        template="simple_white",
        xaxis_range=[0, 15],
        yaxis_range=[0, 15],
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        xaxis_linewidth=3,
        yaxis_linewidth=3,
        xaxis_tickfont_size=20,
        yaxis_tickfont_size=20,
    )
    if filename:
        fig.write_html(filename + ".html")
    else:
        fig.show()


def PlotPredictionError(result: np.array, filename: str = None) -> None:
    """Plot error distribution

    Args:
        result (np.array): the error results comparing to experimental values.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.

    """
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
    levels = ["<= 0.5 eV", "0.5 ~ 1 eV", "> 1 eV"]
    marker_colors = ["green", "orange", "red"]

    fig = go.Figure()
    length = int(len(result) / 3)
    nums = []
    for i in range(length):
        nums.append(Count(result[2 + 3 * i] - result[1 + 3 * i]))
    nums = np.array(nums)

    match length:
        case 4:
            names = names1
        case 6:
            names = names2

    for i in range(3):
        fig.add_trace(
            go.Bar(
                name=levels[i],
                y=names,
                x=nums[:, 3 + i],
                orientation="h",
                marker_color=marker_colors[i],
            )
        )

    # annotations = []
    # for i in range()

    fig.update_layout(
        barmode="stack",
        title="Error Distribution - SVR (rbf, Basis 1)",
        template="simple_white",
        paper_bgcolor="white",
        showlegend=True,
        xaxis_title="Percentage (%)",
        titlefont=dict(size=24),
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        xaxis_linewidth=3,
        yaxis_linewidth=3,
        xaxis_tickfont_size=20,
        yaxis_tickfont_size=20,
        width=1300,
        height=800,
    )
    if filename:
        fig.write_html(f"{filename}.html")
    else:
        fig.show()


def PlotPredictionMAPE(result: np.array, filename: str = None) -> None:
    """Plot MAPE distribution of prediction result

    Args:
        result (np.array): the prediction result.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.

    """
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

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

    # start with a rectangular Figure
    fig = go.Figure()
    length = int(len(result) / 3)

    match length:
        case 4:
            for i in range(4):
                fig.add_trace(
                    go.Scatter(
                        x=result[1 + 3 * i],
                        y=(result[2 + 3 * i] / result[1 + 3 * i]) - 1,
                        hovertext=result[3 * i],
                        mode="markers",
                        name=names1[i],
                        marker=dict(size=14, line=dict(width=2, color="black")),
                    )
                )

    fig.add_hline(y=0, line=dict(width=3, dash="dash", color="red"))

    fig.update_layout(
        title="Error Distribution Range - SVR (rbf-Basis 1)",
        template="simple_white",
        paper_bgcolor="white",
        showlegend=True,
        xaxis_title="E_exp. (eV)",
        yaxis_title="E_calc./E_exp. - 1",
        titlefont=dict(size=24),
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        xaxis_linewidth=3,
        yaxis_linewidth=3,
        xaxis_tickfont_size=20,
        yaxis_tickfont_size=20,
        width=1300,
        height=800,
    )
    if filename:
        fig.write_html(f"{filename}.html")
    else:
        fig.show()


def OldPlotPrediction(result: np.array):
    y_train = result[1]
    y_train_pred = result[2]
    y_validation = result[4]
    y_validation_pred = result[5]
    y_majority_test = result[7]
    y_majority_test_pred = result[8]
    y_minority_test = result[10]
    y_minority_test_pred = result[11]

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.xlabel("Experimental Values (eV)")
    plt.ylabel("Predicted Values (eV)")
    l00 = plt.plot([0, 14], [0, 14], linestyle="--", color="red")
    l01 = plt.plot([5, 5], [0, 14], linestyle="--", color="black")
    l1 = plt.scatter(y_train, y_train_pred, marker=".", color="#1f77b4")
    l2 = plt.scatter(y_validation, y_validation_pred, marker="x", color="#ff7f0e")
    l3 = plt.scatter(y_majority_test, y_majority_test_pred, marker="v", color="#2ca02c")
    l4 = plt.scatter(y_minority_test, y_minority_test_pred, marker="^", color="#d62728")
    plt.legend(
        handles=[l1, l2, l3, l4],
        labels=[
            "Training(Majority)",
            "Validation(Majority)",
            "Test(Majority)",
            "Test(Minority)",
        ],
        loc="best",
    )
    plt.title("Basis 1 - SVR")
    plt.show()


def OldPlotPredictionError(result: np.array):
    y_train_d = result[2] - result[1]
    y_validation_d = result[5] - result[4]
    y_majority_test_d = result[8] - result[7]
    y_minority_test_d = result[11] - result[10]
    nums = []
    nums.append(Count(y_train_d))
    nums.append(Count(y_validation_d))
    nums.append(Count(y_majority_test_d))
    nums.append(Count(y_minority_test_d))
    nums = np.array(nums)

    plt.rcParams["figure.figsize"] = (5, 3)
    fig, ax = plt.subplots(constrained_layout=True)
    c1 = nums[:, 3]
    c2 = nums[:, 4]
    c3 = nums[:, 5]

    ind = np.arange(4)
    width = 0.45
    p1 = ax.barh(ind, c1, width, color="green", label="<= 0.5 eV")
    p2 = ax.barh(ind, c2, width, color="orange", left=c1, label="0.5~1 eV")
    p3 = ax.barh(ind, c3, width, color="red", left=c1 + c2, label="> 1 eV")

    ax.bar_label(p1, label_type="center")
    ax.bar_label(p2, label_type="center")
    ax.bar_label(p3, label_type="center")
    ax.set_yticks(
        ind,
        labels=[
            "Majority Train",
            "Majority Validaiton",
            "Majority Test",
            "Minority Test",
        ],
    )
    ax.invert_yaxis
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Error Distribution - SVR(rbf)")
    ax.legend(loc="upper left")
    plt.show()


def OldPlotPredictionMAPE(result: np.array):
    x1 = result[1]
    y1 = (result[2] / result[1]) - 1
    x2 = result[4]
    y2 = (result[5] / result[4]) - 1
    x3 = result[7]
    y3 = (result[8] / result[7]) - 1
    x4 = result[10]
    y4 = (result[11] / result[10]) - 1

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 5))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction="in", top=True, right=True, grid_alpha=0.5)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction="in", labelleft=False, grid_alpha=0.5)

    # the scatter plot:
    ax_scatter.scatter(x1, y1, alpha=0.5, label="Train(Majority)")
    ax_scatter.scatter(x2, y2, alpha=0.5, label="Validation(Majority)")
    ax_scatter.scatter(x3, y3, alpha=0.5, label="Test(Majority)")
    ax_scatter.scatter(x4, y4, alpha=0.5, label="Test(Minority)")
    ax_scatter.grid(True, linestyle="--")
    ax_scatter.axhline(0, linestyle="--", color="r")

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(
        np.max(np.abs(y1)), np.max(np.abs(y2)), np.max(np.abs(y3)), np.max(np.abs(y4))
    )
    ymax = max(
        np.max(np.abs(y1)), np.max(np.abs(y2)), np.max(np.abs(y3)), np.max(np.abs(y4))
    )
    lim = (int(xymax / binwidth) + 1) * binwidth

    ax_scatter.set_xlim((0, 23))
    ax_scatter.set_ylim((-1.0, 1.0))
    # ax_scatter.set_ylim((-ymax, ymax))
    ax_scatter.set_xlabel("${E_{exp.}}$ (eV)")
    ax_scatter.set_ylabel("${E_{calc.}/E_{exp.}-1}$")

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy.hist(
        y1,
        bins=bins,
        density=True,
        alpha=0.5,
        histtype="stepfilled",
        orientation="horizontal",
    )
    ax_histy.hist(
        y2,
        bins=bins,
        density=True,
        alpha=0.5,
        histtype="stepfilled",
        orientation="horizontal",
    )
    ax_histy.hist(
        y3,
        bins=bins,
        density=True,
        alpha=0.5,
        histtype="stepfilled",
        orientation="horizontal",
    )
    ax_histy.hist(
        y4,
        bins=bins,
        density=True,
        alpha=0.5,
        histtype="stepfilled",
        orientation="horizontal",
    )
    ax_histy.axhline(0, linestyle="--", color="r")

    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.set_xlabel("Rel. Freq.")
    ax_histy.grid(True, linestyle="--")

    ax_scatter.legend()
    plt.suptitle("Error Distribution Range - SVR(rbf-Basis1)")
    plt.show()

    print("OK")


def OldPlotPredictionPlotly(result: np.array):
    train = pd.DataFrame(result[:3]).T
    train.insert(train.shape[1], "4", "Train(Majority)")
    validation = pd.DataFrame(result[3:6]).T
    validation.insert(validation.shape[1], "4", "Validation(Majority)")
    majority_test = pd.DataFrame(result[6:9]).T
    majority_test.insert(majority_test.shape[1], "4", "Test(Majority)")
    minority_test = pd.DataFrame(result[9:]).T
    minority_test.insert(minority_test.shape[1], "4", "Test(Minority)")
    df = pd.concat([train, validation, majority_test, minority_test])
    df = df.reset_index(drop=True)
    df.columns = ["Materials", "Experimental (eV)", "Prediction (eV)", "Set"]

    fig = px.scatter(
        df,
        x="Experimental (eV)",
        y="Prediction (eV)",
        # colors="Set",
        symbol="Set",
        hover_name="Materials",
    )
    fig.show()


def PlotHyperparametr(
    op_summarys: np.array,
    filename: str,
    is_std: bool = True,
    is_show: bool = False,
    metric: str = "mae",
) -> None:
    """Plot results based on different values of one hyperparameter

    Args:
        op_summarys (np.array): summary data for hyperparameter optimization.
        filename (str): the filename you want to save the figure.
        is_std (bool, optional): whether with standard errors. Defaults to True.
        is_show (bool, optional): whether show immediately. Defaults to False.
        metric (str, optional): the metric used to compare, which can be "rmse" or
            "mae". Defaults to "mae".

    """
    match metric:
        case "mae":
            x = []
            train_mae = []
            train_std = []
            train_maj_mae = []
            train_maj_std = []
            train_min_mae = []
            train_min_std = []
            validation_mae = []
            validation_std = []
            validation_maj_mae = []
            validation_maj_std = []
            validation_min_mae = []
            validation_min_std = []
            test_mae = []
            test_std = []
            test_maj_mae = []
            test_maj_std = []
            test_min_mae = []
            test_min_std = []

            for op_summary in op_summarys:
                x.append(op_summary[0][0])
                train_mae.append(op_summary[1][0])
                train_std.append(op_summary[1][1])
                train_maj_mae.append(op_summary[1][2])
                train_maj_std.append(op_summary[1][3])
                train_min_mae.append(op_summary[1][4])
                train_min_std.append(op_summary[1][5])
                validation_mae.append(op_summary[1][10])
                validation_std.append(op_summary[1][11])
                validation_maj_mae.append(op_summary[1][12])
                validation_maj_std.append(op_summary[1][13])
                validation_min_mae.append(op_summary[1][14])
                validation_min_std.append(op_summary[1][15])
                test_mae.append(op_summary[1][20])
                test_std.append(op_summary[1][21])
                test_maj_mae.append(op_summary[1][22])
                test_maj_std.append(op_summary[1][23])
                test_min_mae.append(op_summary[1][24])
                test_min_std.append(op_summary[1][25])

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    name="Training",
                    y=train_mae,
                    x=x,
                    error_y=dict(type="data", array=train_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation",
                    y=validation_mae,
                    x=x,
                    error_y=dict(type="data", array=validation_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test",
                    y=test_mae,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Training (Majority)",
                    y=train_maj_mae,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation (Majority)",
                    y=validation_maj_mae,
                    x=x,
                    error_y=dict(type="data", array=validation_maj_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test (Majority)",
                    y=test_maj_mae,
                    x=x,
                    error_y=dict(type="data", array=test_maj_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Training (Minority)",
                    y=train_min_mae,
                    x=x,
                    error_y=dict(type="data", array=train_min_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation (Minority)",
                    y=validation_min_mae,
                    x=x,
                    error_y=dict(type="data", array=validation_min_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test (Minority)",
                    y=test_min_mae,
                    x=x,
                    error_y=dict(type="data", array=test_min_std, visible=is_std),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.update_xaxes(type="log")
            fig.update_layout(
                title="The influence of hyperparameter C",
                template="simple_white",
                paper_bgcolor="white",
                showlegend=True,
                xaxis_title="Hyperparameter C",
                yaxis_title="MAE (eV)",
                titlefont=dict(size=24),
                xaxis_title_font_size=24,
                yaxis_title_font_size=24,
                xaxis_linewidth=3,
                yaxis_linewidth=3,
                xaxis_tickfont_size=20,
                yaxis_tickfont_size=20,
                width=1300,
                height=800,
            )

            if is_show:
                fig.show()
            if is_std:
                fig.write_html(f"{filename}_mae.html")
            else:
                fig.write_html(f"{filename}_mae_nostd.html")

        case "rmse":
            x = []
            train_rmse = []
            train_maj_rmse = []
            train_min_rmse = []
            validation_rmse = []
            validation_maj_rmse = []
            validation_min_rmse = []
            test_rmse = []
            test_maj_rmse = []
            test_min_rmse = []

            for op_summary in op_summarys:
                x.append(op_summary[0][0])
                train_rmse.append(op_summary[1][7])
                train_maj_rmse.append(op_summary[1][30])
                train_min_rmse.append(op_summary[1][31])
                validation_rmse.append(op_summary[1][17])
                validation_maj_rmse.append(op_summary[1][32])
                validation_min_rmse.append(op_summary[1][33])
                test_rmse.append(op_summary[1][27])
                test_maj_rmse.append(op_summary[1][34])
                test_min_rmse.append(op_summary[1][35])

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    name="Training",
                    y=train_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation",
                    y=validation_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test",
                    y=test_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Training (Majority)",
                    y=train_maj_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation (Majority)",
                    y=validation_maj_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test (Majority)",
                    y=test_maj_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Training (Minority)",
                    y=train_min_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Validation (Minority)",
                    y=validation_min_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Test (Minority)",
                    y=test_min_rmse,
                    x=x,
                    error_y=dict(type="data"),
                    mode="lines+markers",
                    marker_size=10,
                )
            )

            fig.update_xaxes(type="log")
            fig.update_layout(
                title="The influence of hyperparameter C",
                template="simple_white",
                paper_bgcolor="white",
                showlegend=True,
                xaxis_title="Hyperparameter C",
                yaxis_title="RMSE (eV)",
                titlefont=dict(size=24),
                xaxis_title_font_size=24,
                yaxis_title_font_size=24,
                xaxis_linewidth=3,
                yaxis_linewidth=3,
                xaxis_tickfont_size=20,
                yaxis_tickfont_size=20,
                width=1300,
                height=800,
            )

            if is_show:
                fig.show()

            fig.write_html(f"{filename}_rmse.html")