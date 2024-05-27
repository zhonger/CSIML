"""Plot figures for evaluating errors"""

import numpy as np
import plotly.graph_objects as go

from ._base import count_errors, get_names, plotly_post


def plot_prediction_error(
    result: np.array, method: str, filename: str = None, **kws
) -> None:
    """Plot error distribution

    Args:
        result (np.array): the error results comparing to experimental values.
        method (str): the method name.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
    """
    names = get_names(method)
    levels = ["<= 0.5 eV", "0.5 ~ 1 eV", "> 1 eV"]
    marker_colors = kws.get("marker_colors", ["green", "orange", "red"])
    nums = []
    for i in range(int(len(result) / 3)):
        nums.append(count_errors(result[2 + 3 * i] - result[1 + 3 * i]))
    nums = np.array(nums)

    fig = go.Figure()
    for i in range(3):
        kwr = {"name": levels[i], "orientation": "h", "marker_color": marker_colors[i]}
        fig.add_trace(go.Bar(x=nums[:, 3 + i], y=names, **kwr))
    fig.update_layout(
        barmode="stack",
        title=f"Error Distribution - {method}",
        template="simple_white",
        paper_bgcolor="white",
        showlegend=True,
        xaxis_title="Percentage (%)",
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
    plotly_post(fig, filename)


def plot_prediction_mape(
    result: np.array, method: str, metric: str = "mape", filename: str = None
) -> None:
    """Plot MAPE distribution of prediction result

    Args:
        result (np.array): the prediction result.
        method (str): the method name.
        filename (str, optional): the filename you want to save the figure. Defaults to
            None.
    """
    names = get_names(method)

    fig = go.Figure()
    marker = {"size": 14, "line": {"width": 2, "color": "black"}}
    for i in range(int(len(result) / 3)):
        kwr = {
            "hovertext": result[3 * i],
            "mode": "markers",
            "name": names[i],
            "marker": marker,
        }
        x = result[1 + 3 * i]
        if metric == "mape":
            y = (result[2 + 3 * i] / result[1 + 3 * i]) - 1
        else:
            y = result[2 + 3 * i] - result[1 + 3 * i]
        fig.add_trace(go.Scatter(x=x, y=y, **kwr))

    hline = {"width": 3, "dash": "dash", "color": "red"}
    fig.add_hline(y=0, line=hline)

    fig.update_layout(
        title=f"Error Distribution Range - {method}",
        template="simple_white",
        paper_bgcolor="white",
        showlegend=True,
        xaxis_title=r"$E_{exp.} (eV)$",
        yaxis_title=r"$E_{calc.}/E_{exp.} - 1$",
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
    plotly_post(fig, filename)
