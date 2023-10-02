"""Weights analysis module"""

import numpy as np
import pandas as pd

from .main import save_analyzed_result


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
    raise NameError("Two results are not match! Please check them firstly!")
