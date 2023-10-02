"""Plot figures for evaluating performance"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

from ._base import get_names, plot_post, plotly_post


def plot_prediction(
    result: np.array, method: str, title: str = None, filename: str = None, **kws
) -> None:
    """Plot prediction-experimental values

    Args:
        result (np.array): prediction result.
        method (str): the method name.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
    """
    names = get_names(method)
    colors = kws.get(
        "colors",
        [
            "rgb(31, 119, 180)",
            "rgb(255, 127, 13)",
            "rgb(44, 160, 44)",
            "rgb(214, 38, 40)",
            "rgb(148, 103, 189)",
            "rgb(140, 86, 75)",
        ],
    )
    if title is None:
        title = method

    fig = go.Figure()
    plot_prediction_bg_error(fig)
    plot_prediction_points(fig, result, names, colors)
    plot_prediction_bg(fig)
    fig.update_layout(
        title="<b>" + title + "<b>",
        titlefont={"size": 24},
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
    plotly_post(fig, filename)


def plot_prediction_points(
    fig: go.Figure, result: list, names: list, colors: list
) -> None:
    length = 6 if int(len(result) / 3) > 6 else int(len(result) / 3)
    for i in range(length):
        marker = {
            "size": 14,
            "color": colors[i],
            "line": {"width": 2, "color": "black"},
        }
        kws = {
            "hovertext": result[3 * i],
            "mode": "markers",
            "name": names[i],
            "marker": marker,
        }
        fig.add_trace(go.Scatter(x=result[1 + 3 * i], y=result[2 + 3 * i], **kws))


def plot_prediction_bg(fig: go.Figure) -> None:
    kws = {
        "annotation_position": "top",
        "annotation_font_size": 24,
        "opacity": 0.5,
        "layer": "below",
        "line_width": 0,
    }
    redline = {"width": 3, "dash": "dash", "color": "red"}
    fig.add_vline(x=5, line=redline)
    fig.add_vrect(x0=0, x1=5, annotation_text="Majority", fillcolor="skyblue", **kws)
    fig.add_vrect(x0=5, x1=15, annotation_text="Minority", fillcolor="lightgray", **kws)


def plot_prediction_bg_error(fig: go.Figure) -> None:
    whiteline = {"color": "rgba(255,255,255,0)"}
    errorline1 = {
        "fill": "toself",
        "fillcolor": "rgba(226, 201, 212, 1)",
        "line": whiteline,
        "name": "1 eV",
    }
    errorline2 = {
        "fill": "toself",
        "fillcolor": "rgba(204, 219, 214, 1)",
        "line": whiteline,
        "name": "0.5 eV",
    }
    bestline = {"marker_color": "red", "line_dash": "dash", "name": "Best line"}
    fig.add_trace(
        go.Scatter(x=[-3, 16] + [16, -3], y=[-2, 17] + [15, -4], **errorline1)
    )
    fig.add_trace(
        go.Scatter(x=[-3, 16] + [16, -3], y=[-2.5, 16.5] + [15.5, -3.5], **errorline2)
    )
    fig.add_trace(go.Scatter(x=[-3, 16], y=[-3, 16], **bestline))


def plot_rider(y: list, filename: str = None, **kws) -> None:
    dpi = kws.get("dpi", 100)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], aspect=1)

    size = 0.1
    vals = np.ones(12)

    # A nice set of colors for seasons
    cmap20c = plt.get_cmap("tab20c")
    cmap20b = plt.get_cmap("tab20b")
    colors = [
        cmap20c(0),
        cmap20c(1),
        cmap20c(2),  # Winter
        cmap20c(10),
        cmap20c(9),
        cmap20c(8),  # Spring
        cmap20c(4),
        cmap20c(5),
        cmap20c(6),  # Summer
        cmap20b(15),
        cmap20b(14),
        cmap20b(13),  # Autumn
    ]

    # Simple pie
    ax.pie(
        np.ones(12),
        radius=1.0,
        colors=colors,
        wedgeprops=dict(width=size, edgecolor="w"),
    )

    labels = [
        "Total MAE",
        "Total MAE STD",
        "Majority MAE",
        "Majority MAE STD",
        "Minority MAE",
        "Minority MAE STD",
        "Vairance",
        "Total RMSE",
        "Majority RMSE",
        "Minority RMSE",
        "MAPE",
        r"$1-R^2$",
    ]
    for i, label in enumerate(labels):
        plot_rider_label(ax, label, (i+0.5)*2*np.pi/12, 1-0.5*size)
    # This could be made through a list but it is easier to red this way
    # plot_rider_label(ax, "Total MAE", 0.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Majority MAE", 1.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Minority MAE", 2.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "MAE", 1.5 * 2 * np.pi / 12, 1 + size, 0.0125)

    # plot_rider_label(ax, "Minority RMSE", 3.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Majority RMSE", 4.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Total RMSE", 5.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "RMSE", 4.5 * 2 * np.pi / 12, 1 + size, 0.0125)

    # plot_rider_label(ax, "Variance", 6.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "MAPE", 7.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, r"$1-R^2$", 8.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Others", 7.5 * 2 * np.pi / 12, 1 + size, 0.0125)

    # plot_rider_label(ax, "Minority MAE STD", 9.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Majority MAE STD", 10.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "Total MAE STD", 11.5 * 2 * np.pi / 12, 1 - 0.5 * size)
    # plot_rider_label(ax, "MAE STD", 10.5 * 2 * np.pi / 12, 1 + size, 0.0125)

    # Add a polar projection on top of the previous one
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7], projection="polar")

    # ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False)+1/12*np.pi)
    # ax.set_yticks(np.linspace(0, 1, 6))
    x = np.linspace(0, 2 * np.pi, 12, endpoint=False) + 1 / 12 * np.pi
    # y = [1, 0.8, 1, 0.2, 0.5, 0.4, 0.3, 0.2, 0.9, 0.8, 0.7, 1]
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    ax.plot(x, y, linewidth=3, color="steelblue")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([])
    ax.set_yticks([1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_rorigin(-0.25)
    ts = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for t in ts:
        ax.vlines(t, 0, 1.2, linestyles="--", linewidth=1, colors="black")
    ax.hlines(1.0, 0, 2 * np.pi, linewidth=3, colors="black")
    ax.hlines(y[0], 0, 2 * np.pi, linewidth=3, colors="black")
    ax.tick_params(width=3)
    ax.fill_between(x, y, facecolor="lightblue")
    plot_post(filename, dpi)


def plot_rider_label(ax, text, angle, radius=1.0, scale=0.005):
    fp = FontProperties(weight="bold")
    path = TextPath((0, 0), text, size=10, prop=fp)
    path.vertices.flags.writeable = True
    V = path.vertices
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    V -= (xmin + xmax) / 2, (ymin + ymax) / 2
    V *= scale
    for i in range(len(V)):
        a = angle - V[i, 0]
        V[i, 0] = (radius + V[i, 1]) * np.cos(a)
        V[i, 1] = (radius + V[i, 1]) * np.sin(a)
    patch = PathPatch(path, facecolor="k", linewidth=0)
    ax.add_artist(patch)
