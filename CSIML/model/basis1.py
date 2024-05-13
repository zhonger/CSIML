"""The module for basis1 method"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from CSIML.analysis.analysis import index_to_name
from CSIML.utils.timer import timer

from ._base import BaseModel


class Basis1Model(BaseModel):
    """Basis1 method, training only with majority set

    Args:
        data (pd.DataFrame): the dataset.
        basic_model (str, optional): the basic ML model. Defaults to "SVR".

    Attributes:
        cv_method (str): Defaults to "basis1". It cannot be changed.

    Other supported parameters please refer to :class:`BaseModel`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        basic_model: str = "SVR",
        **kws,
    ) -> None:
        cv_method = "basis1"
        super().__init__(data, basic_model, cv_method, **kws)

    def fit(self) -> list:
        """Train ML models only with majority set

        Returns:
            list: return trained ML models.
        """
        splited_data = self.splited_data
        X, y = self.X, self.y
        regr = self.regr
        iterations = self.iterations
        regrs = []

        pbar = tqdm(total=iterations)
        pbar.set_description("Basis 1 (Fit)")
        for dataset in splited_data:
            train = dataset[0]
            X_train, y_train = X[train, :], y[train]
            regr = regr.fit(X_train, y_train)
            regrs.append(regr)
            pbar.update(1)

        return regrs

    def predict(self, regrs: list) -> list:
        """Predict all instances in each cross validation iteration with trained ML models

        Args:
            regrs (list): trained machine learning models.

        Returns:
            list: return prediction results.
        """
        data = self.data
        splited_data = self.splited_data
        X = self.X
        y = self.y
        results = []

        pbar = tqdm(total=self.iterations)
        pbar.set_description("Basis 1 (Predict)")
        for dataset, regr in zip(splited_data, regrs):
            train, validation, majority_test, minority_test = dataset
            majority_test_num = len(majority_test)
            test = np.append(majority_test, minority_test)
            X_train, y_train = X[train, :], y[train]
            X_validation, y_validation = X[validation, :], y[validation]
            X_test, y_test = X[test, :], y[test]
            y_majority_test = y_test[:majority_test_num]
            y_minority_test = y_test[majority_test_num:]
            y_train_pred = regr.predict(X_train)
            y_validation_pred = regr.predict(X_validation)
            y_test_pred = regr.predict(X_test)
            y_majority_test_pred = y_test_pred[:majority_test_num]
            y_minority_test_pred = y_test_pred[majority_test_num:]

            # Update index to materials name
            train = index_to_name(data, train)
            validation = index_to_name(data, validation)
            majority_test = index_to_name(data, majority_test)
            minority_test = index_to_name(data, minority_test)

            result = (
                [train]
                + [y_train]
                + [y_train_pred]
                + [validation]
                + [y_validation]
                + [y_validation_pred]
                + [majority_test]
                + [y_majority_test]
                + [y_majority_test_pred]
                + [minority_test]
                + [y_minority_test]
                + [y_minority_test_pred]
            )

            results.append(result)
            pbar.update(1)

        return results

    @timer
    def fit_predict(self) -> list:
        """Train and predict for all instances in each cross validaton iteration

        Returns:
            list: return prediction results.
        """
        regrs = self.fit()
        results = self.predict(regrs)
        return results
