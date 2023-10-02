"""Plot figures for evaluating hyperparameters"""

import numpy as np
import plotly.graph_objects as go


def plot_hyperparmeter(
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
                x.append(op_summary[0])
                train_mae.append(op_summary[1][0])
                train_std.append(op_summary[1][1])
                train_maj_mae.append(op_summary[1][2])
                train_maj_std.append(op_summary[1][3])
                train_min_mae.append(op_summary[1][4])
                train_min_std.append(op_summary[1][5])
                validation_mae.append(op_summary[1][12])
                validation_std.append(op_summary[1][13])
                validation_maj_mae.append(op_summary[1][14])
                validation_maj_std.append(op_summary[1][15])
                validation_min_mae.append(op_summary[1][16])
                validation_min_std.append(op_summary[1][17])
                test_mae.append(op_summary[1][24])
                test_std.append(op_summary[1][25])
                test_maj_mae.append(op_summary[1][26])
                test_maj_std.append(op_summary[1][27])
                test_min_mae.append(op_summary[1][28])
                test_min_std.append(op_summary[1][29])

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
                x.append(op_summary[0])
                train_rmse.append(op_summary[1][7])
                train_maj_rmse.append(op_summary[1][8])
                train_min_rmse.append(op_summary[1][9])
                validation_rmse.append(op_summary[1][19])
                validation_maj_rmse.append(op_summary[1][20])
                validation_min_rmse.append(op_summary[1][21])
                test_rmse.append(op_summary[1][31])
                test_maj_rmse.append(op_summary[1][32])
                test_min_rmse.append(op_summary[1][33])

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
