"""The module for basic model"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor

from CSIML.cross_validation.cv import DatasetSize


class BaseModel(DatasetSize):
    """Base machine learning model

    It supports:
        * "SVR" (the same with "NuSVR")
        * "DT" (Decision Tree)
        * "RF" (Random Forest)
        * "Ada" (AdaBoost)
        * "MLP" (Multi-layer Perceptron)
        * "NuSVR" (Support Vector Regression with rbf kernel)
        * "lSVR" (Support Vector Regression with linear kernel)
        * "pSVR" (Support Vector Regression with polynomial kernel)

    Args:
        data (pd.DataFrame): the dataset.
        basic_model (str, optional): the basic machine model. Defaults to "SVR".
        cv_method (str, optional): cross validation method, Defaults to "siml".
        sampling_method (str, optional): resampling method, supporting "oversampling",
            "undersampling". Defaults to None.
        threhold (float, optional): for splitting majority and minority. Defaults to 5.0
            (for bandgaps).
        parameters (dict, optional): parameters for basic machine learning models.

    Attributes:
        regr: the basic machine learning model.
        X (np.array): the features (the input of the model).
        y (np.array): the property (the output of the model).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        basic_model: str = "SVR",
        cv_method: str = "siml",
        **kws,
    ) -> None:
        self.basic_model = basic_model
        match basic_model := self.basic_model:
            case "custom":
                if "regr" in kws:
                    regr = kws["regr"]
                else:
                    raise KeyError(
                        "'regr' should be defined when basic model is customized."
                    )
            case "DT":
                regr = DecisionTreeRegressor(max_depth=10)
            case "RF":
                regr = RandomForestRegressor(n_estimators=10)
            case "Ada":
                regr = AdaBoostRegressor(n_estimators=10)
            case "MLP":
                regr = MLPRegressor(random_state=1, max_iter=500)
            case "NuSVR":
                regr = NuSVR(kernel="rbf", C=10)
            case "lSVR":
                regr = SVR(kernel="linear", C=10)
            case "pSVR":
                regr = SVR(kernel="poly", C=10)
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

        Args:
            ascending (bool, optional): whether ascending by the property value or not.
                Defaults to False.
            normalize (bool, optional): whether using MinMax Normalization. Defauls to
                True.
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
