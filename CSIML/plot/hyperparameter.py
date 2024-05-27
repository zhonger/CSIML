"""Plot figures for evaluating hyperparameters"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from ._base import plot_post, plotly_post
from ._dataclass import Metrics


def plot_hyperparameter(op_summarys: np.array, **kws) -> None:
    """Plot results based on different values of one hyperparameter

    Args:
        op_summarys (np.array): summary data for hyperparameter optimization.
        filename (str, optionall): the filename you want to save the figure. Defaults to
            None.
        is_std (bool, optional): whether with standard errors. Defaults to True.
        metric (str, optional): the metric used to compare, which can be "rmse" or
            "mae". Defaults to "mae".
    """
    x, train, validation, test = get_hyperparameter_metrics(op_summarys, **kws)
    names = [["Training", "Validation", "Test"], ["", " (Majoirty)", " (Minority)"]]
    fig = go.Figure()
    i = 0
    for values in (train, validation, test):
        j = 0
        for value in (values.total, values.maj, values.min):
            scatter_kws = {
                "name": f"{names[0][i]}{names[1][j]}",
                "mode": "lines+markers",
                "marker_size": 10,
            }
            if kws.get("metric", "mae") == "mae":
                scatter_kws["error_y"] = {
                    "type": "data",
                    "array": value.std,
                    "visible": kws.get("is_std", True),
                }
            fig.add_trace(go.Scatter(x=x, y=value.error, **scatter_kws))
            j += 1
        i += 1
    fig.update_xaxes(type="log")
    fig.update_layout(
        title="The influence of hyperparameter C",
        template="simple_white",
        paper_bgcolor="white",
        showlegend=True,
        xaxis_title="Hyperparameter C",
        yaxis_title="MAE (eV)" if kws.get("metric", "mae") == "mae" else "RMSE (eV)",
        titlefont={"size": 24},
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        xaxis_linewidth=3,
        yaxis_linewidth=3,
        xaxis_tickfont_size=20,
        yaxis_tickfont_size=20,
        width=1300,
        height=800,
    )
    if filename := kws.get("filename", None):
        if kws.get("is_std", True):
            filename = f"{filename}_mae.html"
        else:
            filename = f"{filename}_mae_nostd.html"
    plotly_post(fig, filename)


def get_hyperparameter_metrics(
    op_summarys: list, **kws
) -> tuple[list, Metrics, Metrics, Metrics]:
    """Get metrics from hyperparameter summary data

    Args:
        op_summarys (list): summary data.
        metric (str, optional): the metric used to compare, which can be "rmse" or "mae".
            Defaults to "mae".

    Returns:
        tuple[list, Metrics, Metrics, Metrics]: hyperparameters and the perforamnce in
            training, validation and test set.
    """
    metric = kws.get("metric", "mae")
    x = []
    train = Metrics()
    validation = Metrics()
    test = Metrics()
    for op_summary in op_summarys:
        x.append(op_summary[0])
        if metric == "mae":
            train.total.error.append(op_summary[1][0])
            train.total.std.append(op_summary[1][1])
            train.maj.error.append(op_summary[1][2])
            train.maj.std.append(op_summary[1][3])
            train.min.error.append(op_summary[1][4])
            train.min.std.append(op_summary[1][5])
            validation.total.error.append(op_summary[1][12])
            validation.total.std.append(op_summary[1][13])
            validation.maj.error.append(op_summary[1][14])
            validation.maj.std.append(op_summary[1][15])
            validation.min.error.append(op_summary[1][16])
            validation.min.std.append(op_summary[1][17])
            test.total.error.append(op_summary[1][24])
            test.total.std.append(op_summary[1][25])
            test.maj.error.append(op_summary[1][26])
            test.maj.std.append(op_summary[1][27])
            test.min.error.append(op_summary[1][28])
            test.min.std.append(op_summary[1][29])
        else:
            train.total.error.append(op_summary[1][7])
            train.maj.error.append(op_summary[1][8])
            train.min.error.append(op_summary[1][9])
            validation.total.error.append(op_summary[1][19])
            validation.maj.error.append(op_summary[1][20])
            validation.min.error.append(op_summary[1][21])
            test.total.error.append(op_summary[1][31])
            test.maj.error.append(op_summary[1][32])
            test.min.error.append(op_summary[1][33])
    return x, train, validation, test


def get_hyper_results(azs: list, hypers: list, indexes: list, names: list)->pd.DataFrame:
    """Get results for hyperparameter optimization

    Args:
        azs (list): analysis results.
        hypers (list): hyperparameters.
        indexes (list): the metric indexes in all metrics.
        names (list): the names for hyperparameter optimization metrics.

    Returns:
        pd.DataFrame: return the required metrics for hyperparameter optimization.
    """
    results = []
    for az, hyper in zip(azs, hypers):
        for s in az.summarys:
            for i, n in zip(indexes, names):
                result = []
                result.append(hyper)
                result.append(s[i])
                result.append(n)
                results.append(result)
    columns = ["hyper", "RMSE", "name"]
    hdata = pd.DataFrame(results, columns=columns)
    return hdata


def get_hyper_results_batch(azs: list, hypers: list = range(2, 11))->list:
    """Get results for hyperparameter optimization in batches

    Args:
        azs (list): analysis results.
        hypers (list, optional): hyperparameters. Defaults to range(2, 11).

    Returns:
        list: return the results for hyperparameter optimization plotting.
    """
    indexes = [
        [7, 19],
        [8, 9, 20, 21],
        [31, 32, 33],
    ]
    names = [
        ["Training Total", "Validation Total"],
        [
            "Training Majority",
            "Training Minority",
            "Validation Majority",
            "Validation Minority",
        ],
        ["Test Total", "Test Majority", "Test Minority"],
    ]
    data = []
    for index, name in zip(indexes, names):
        hdata = get_hyper_results(azs, hypers, index, name)
        data.append(hdata)
    return data


def plot_hyper(hdata: list, labels: list, filename: str = None, **kws):
    """Plot hyperparameter optimization results from one view

    Args:
        hdata (list): the results for hyperparameter optimization.
        labels (list): the labels for point kinds.
        filename (str, optional): the filename to save figure. Defaults to None.
    """
    dpi = kws.get("dpi", 100)
    sns.set_theme(style="ticks", rc={"figure.dpi": dpi})

    if len(labels) == 4:
        palette = ["red", "green"]
    else:
        palette = None
    sns.lineplot(
        x="hyper",
        y="RMSE",
        hue="name",
        style="name",
        data=hdata,
        markers=["o"] * int(len(labels) / 2),
        dashes=False,
        palette=palette,
    )
    plt.legend(labels=labels, ncols=2, fontsize=8)
    if "y_lim" in kws:
        plt.ylim(kws["y_lim"])
    plt.xlabel("Hyperparameter")
    plt.ylabel("RMSE (K)")
    plt.title(kws.get("title"))
    if "x_scale" in kws:
        plt.xscale(kws["x_scale"])
    plot_post(filename, dpi)


def plot_hyper_batch(data: list, path: str = None, **kws):
    """Plot hyperparameter optimization results from different views

    It is plotted from three views:
    - Total view: training total and validation total
    - Tradeoff view: training and validation of three sets (same with below)
    - Test view: test of three sets (total, majority and minority)

    Args:
        data (list): the results for hyperparameter optimization.
        path (str, optional): the path for saving figures. If none, only show figures.
        x_scale (str, optional): the xaxis scale, for maplotlib.pyplot.xscale().
        y_lim (list, optional): the yaxis range.
    """
    labels = [
        [
            "Training Total",
            "Training Total Std",
            "Validation Total",
            "Validation Total Std",
        ],
        [
            "Training Majority",
            "Training Majority Std",
            "Training Minority",
            "Training Minority Std",
            "Validation Majority",
            "Validation Majority Std",
            "Validation Minority",
            "Validation Minority Std",
        ],
        [
            "Test Total",
            "Test Total Std",
            "Test Majority",
            "Test Majority Std",
            "Test Minority",
            "Test Minority Std",
        ],
    ]
    names = ["total", "tradeoff", "test"]
    titles = ["Total RMSE", "Trade-off RMSE", "Test RMSE"]
    for hdata, label, name, title in zip(data, labels, names, titles):
        if path:
            kws["filename"] = f"{path}{name}.png"
        kws["title"] = title
        plot_hyper(hdata, label, **kws)


def plot_hyper_heatmap(
    results: list, n: int, title: str, filename: str = None, **kws
) -> None:
    """Plot hyperparameter optimization results by heatmap

    Args:
        results (list): hyperparameter rmse results.
        n (int): the index in hyperparameters.
        title (str): figure title.
        filename (str, optional):the filename to save. Defaults to None.
    """
    data = pd.DataFrame(
        np.array(results)[:, [-2, -1, n]], columns=["degree", "C", "metric"]
    )
    degrees = data.degree.unique().astype("int")
    hypers = data.C.unique()
    scores = data.pivot(index="C", columns="degree", values="metric")
    rects = []
    for i in range(scores.shape[1]):
        rects.append((i, np.argwhere(scores.index == scores.idxmin().iloc[i])[0, 0]))
    dpi = kws.get("dpi", 100)
    figsize = kws.get("figsize", (15, 10))
    sns.set_theme(rc={"figure.figsize": figsize, "figure.dpi": dpi})
    sns.set_style({"font.family": kws.get("fontfamily", "Times New Roman Cyr")})
    ht = sns.heatmap(
        scores,
        xticklabels=degrees,
        yticklabels=hypers,
        vmax=kws.get("vmax", 600),
        square=True,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8},
    )
    ht.set_title(title, fontdict={"fontweight": "bold"})
    for rect in rects:
        ht.add_patch(Rectangle(rect, 1, 1, fill=False, edgecolor="green", lw=2))
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    ht.set_xlabel("Degree", fontweight="bold")
    ht.set_ylabel("C", fontweight="bold")
    plot_post(filename, dpi)


def plot_hyper_heatmap_batch(results: list, path: str = None, **kws) -> None:
    """Batch plot hyperparameter optimization results by heatmap

    Args:
        results (list): hyperparameter rmse results.
        path (str, optional): the path to save file. Defaults to None.
    """
    numbers = [7, 8, 9, 19, 20, 21, 31, 32, 33]
    titles = [
        "Total Training RMSE",
        "Majority Training RMSE",
        "Minority Training RMSE",
        "Total Valiation RMSE",
        "Majority Validation RMSE",
        "Minority Validation RMSE",
        "Total Test RMSE",
        "Majority Test RMSE",
        "Minority Test RMSE",
    ]
    for n, title in zip(numbers, titles):
        if path:
            filename = "_".join(title.lower().split(" "))
            filename = f"{path}{filename}"
        else:
            filename = None
        plot_hyper_heatmap(results, n, title, filename, **kws)
