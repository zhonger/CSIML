"""The module for basic model"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from SIML.cross_validation.cv import DatasetSize


class BaseModel(DatasetSize):
    def __init__(
        self,
        data: pd.DataFrame,
        basic_model: str = "SVR",
        cv_method: str = "siml",
        **kws,
    ) -> None:
        self.basic_model = basic_model
        match basic_model := self.basic_model:
            case "DT":
                regr = DecisionTreeRegressor(max_depth=10)
            case "RF":
                regr = RandomForestRegressor(n_estimators=10)
            case "Ada":
                regr = AdaBoostRegressor(n_estimators=10)
            case "NuSVR":
                regr = NuSVR(kernel="rbf", C=10)
            case _:
                regr = SVR(kernel="rbf", C=10)
        sampling_method = kws.get("sampling_method", None)
        threshold = kws.get("threshold", 5.0)
        super().__init__(data, cv_method, sampling_method, threshold)

        # Handle parameters for basic model
        self.random_state = kws.get("random_state", 3)
        self.random_state2 = kws.get("random_state2", self.random_state)
        self.parameters = defaultdict(int)
        if "parameters" in kws:
            for k, v in kws["parameters"].items():
                self.parameters[k] = v
        parameters = dict(self.parameters)
        if parameters is not None:
            for k, v in parameters.items():
                regr.__setattr__(k, v)
        self.regr = regr

        self.X = np.array([])
        self.y = np.array([])
        self.cal_size()
        self.split_data(**kws)
        self.preprocessing(**kws)

    def preprocessing(self, **kws) -> None:
        """Data preprocessing

        It mainly includes:
            * reordering the data according to the property value ascendly.
            * filling NaN values with 0.
            * normalizing the features with MinMaxScaler().

        It's optional. For bandgaps with features based on elemetns, it's needed.
        """
        data = self.data
        if kws.get("ascending", False):
            data.sort_values(by="Experimental", inplace=True, ascending=True)
        X = data.iloc[:, 3:].fillna(0).values
        if kws.get("normalize", True):
            X = MinMaxScaler().fit_transform(pd.DataFrame(X))
        y = data.iloc[:, 1].values
        self.X = X
        self.y = y
