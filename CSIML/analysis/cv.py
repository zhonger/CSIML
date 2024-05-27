"""The module for CV method analysis"""

import os
import pickle

import pandas as pd

from CSIML.model.basis2 import Basis2Model

from .main import Analysis


class AnalysisCV:
    """Analysis cross-validation class
    
    Args:
        datasets (list): datasets.
        cv_method (str): the cross-validation method name.
        title (str, optional): the title for the analysis. Defaults to None.
        save_path (str, optional): the path to save files. Defaults to "tmp".
        sampling_method (str, optional): the sampling method. Defaults to None.
        random_state (int, optional): the seed for random sampling. Defaults to 10.
        parameters (dict, optional): the parameters for the basic ML model. Defaults to 
            {"C": 10, "gamma": 0.01, "epsilon": 0.2}.
    """
    def __init__(
        self, path: str, datasets: list, cv_method: str, title: str = None, **kws
    ) -> None:
        self.datasets = datasets
        self.cv_method = cv_method
        self.title = title
        save_path = kws.get("save_path", "tmp")
        sampling_method = kws.get("sampling_method", None)
        random_state = kws.get("random_state", 10)
        parameters = kws.get(
            "parameters",
            {
                "C": 10,
                "gamma": 0.01,
                "epsilon": 0.2,
            },
        )
        summary = []
        for dataset in datasets:
            data = pd.read_excel(path, dataset)
            model = Basis2Model(
                data,
                cv_method=cv_method,
                sampling_method=sampling_method,
                random_state=random_state,
                parameters=parameters,
            )
            print(model)
            results = model.fit_predict()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = f"{save_path}/t{dataset}_{cv_method}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(results, f)
            a1 = Analysis(filename=filename, method=f"{dataset} {title}")
            summary.append(a1)
            print(a1)
        self.summary = summary

    def __str__(self) -> str:
        """The string for print"""
        titles = [
            "Dataset",
            "Total_MAE",
            "Major_MAE",
            "Minor_MAE",
            "Diff",
            "Total_RMSE",
            "Major_RMSE",
            "Minor_RMSE",
            "Diff",
        ]
        msg = ""
        msg = msg.join(f"{title:<12}" for title in titles)
        msg += "\n"
        margin = "------"
        msg += f"{margin:<12}" * len(titles)
        for res, dataset in zip(self.summary, self.datasets):
            msg += (
                f"\n{dataset:<12}"
                f"{res.metrics[24]:<12.3f}"
                f"{res.metrics[26]:<12.3f}"
                f"{res.metrics[28]:<12.3f}"
                f"{res.metrics[28]/res.metrics[26]:<12.3f}"
                f"{res.metrics[31]:<12.3f}"
                f"{res.metrics[32]:<12.3f}"
                f"{res.metrics[33]:<12.3f}"
                f"{res.metrics[33]/res.metrics[32]:<12.3f}"
            )
        return msg
