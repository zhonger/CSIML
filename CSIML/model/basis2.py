"""The module for Basis2 model"""

import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpi4py import MPI
from tqdm.auto import tqdm

if os.getenv("INTELEX", "False") == "True":
    from sklearnex import patch_sklearn

    patch_sklearn()


from CSIML.analysis.analysis import index_to_name
from CSIML.model._base import BaseModel
from CSIML.utils.timer import timer


class Basis2Model(BaseModel):
    """Basis2 method, training with majority and minority set

    Args:
        data (pd.DataFrame): the dataset.
        basic_model (str, optional): the basic machine learning model. Defaults to "SVR".
        cv_method (str, optional): cross validation method. Defaults to "siml".
        mpi_mode (bool, optional): whether using mpi to parallel. Defaults to False.
        show_tips (bool, optional): whether showing progress bar. Defaults to False.
        n_jobs (bool, optional): the number of cores. It only works when `mpi_mode` is
            False. This parallelization is supported by scikit-learn library in single node.

    Other supported parameters please refer to :class:`BaseModel`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        basic_model: str = "SVR",
        cv_method: str = "siml",
        **kws
    ) -> None:
        super().__init__(data, basic_model, cv_method, **kws)
        self.mpi_mode = kws.get("mpi_mode", False)
        self.show_tips = kws.get("show_tips", False)
        self.n_jobs = kws.get("n_jobs", 1)

    def __fit__(self, df: list) -> list:
        """Train single ML model for one iteration of cross validation.

        Args:
            df (list): splitted dataset in one iteration.

        Returns:
            list: return trained ML model.
        """
        regr = self.regr
        train = np.append(df[0], df[3])
        X_train = self.X[train, :]
        y_train = self.y[train]
        regr = regr.fit(X_train, y_train)
        return regr

    def fit(self) -> list:
        """Train ML models with majority and minority set.

        Returns:
            list: return trained ML models.
        """
        splited_data = self.splited_data
        if self.mpi_mode:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            offset = np.linspace(0, self.iterations, size + 1).astype("int")
            local_data = splited_data[offset[rank] : offset[rank + 1]]
            local_regrs = []
            for df in local_data:
                local_regr = self.__fit__(df)
                local_regrs.append(local_regr)
            regrs = comm.allgather(local_regrs)
            regrs = list(itertools.chain(*regrs))
        elif self.n_jobs > 1:
            if self.show_tips:
                regrs = Parallel(self.n_jobs, verbose=3)(
                    delayed(self.__fit__)(splited_data[i])
                    for i in tqdm(range(self.iterations))
                )
            else:
                regrs = Parallel(self.n_jobs, verbose=3)(
                    delayed(self.__fit__)(splited_data[i])
                    for i in range(self.iterations)
                )
        else:
            regrs = []
            if self.show_tips:
                pbar = tqdm(total=self.iterations)
                pbar.set_description("Basis 2 (Fit)")
            for df in splited_data:
                regr = self.__fit__(df)
                regrs.append(regr)
                if self.show_tips:
                    pbar.update(1)
        return regrs

    def __predict__(self, df: list, regr) -> list:
        """Predict result with single trained ML model

        Args:
            df (list): splitted dataset in one iteration.
            regr: single trained ML model.

        Returns:
            list: return prediction result.
        """
        data = self.data
        X = self.X
        y = self.y
        y_pred = regr.predict(X)
        result = []
        for dfi in df:
            result = result + [index_to_name(data, dfi)] + [y[dfi]] + [y_pred[dfi]]
        counts = Counter(df[3])
        result.append(index_to_name(data, list(counts.keys())))
        result.append(np.array(list(counts.values())))
        return result

    def predict(self, regrs: list) -> list:
        """Predict results with trained ML models.

        Args:
            regrs (list): trained ML models.

        Returns:
            list: return prediction results.
        """
        splited_data = self.splited_data
        if self.mpi_mode:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            offset = np.linspace(0, self.iterations, size + 1).astype("int")
            local_data = splited_data[offset[rank] : offset[rank + 1]]
            local_regrs = regrs[offset[rank] : offset[rank + 1]]
            local_results = []
            for df, regr in zip(local_data, local_regrs):
                local_results.append(self.__predict__(df, regr))
            results = comm.allgather(local_results)
            results = list(itertools.chain(*results))
        elif self.n_jobs > 1:
            if self.show_tips:
                results = Parallel(self.n_jobs, verbose=3)(
                    delayed(self.__predict__)(splited_data[i], regrs[i])
                    for i in tqdm(range(self.iterations))
                )
            else:
                results = Parallel(self.n_jobs, verbose=3)(
                    delayed(self.__predict__)(splited_data[i], regrs[i])
                    for i in range(self.iterations)
                )
        else:
            results = []
            if self.show_tips:
                pbar = tqdm(total=self.iterations)
                pbar.set_description("Basis 2 (Predict)")
            for df, regr in zip(splited_data, regrs):
                results.append(self.__predict__(df, regr))
                if self.show_tips:
                    pbar.update(1)
        return results

    @timer
    def fit_predict(self) -> list:
        """Train and predict for all instances

        Returns:
            list: return prediction results.
        """
        regrs = self.fit()
        results = self.predict(regrs)
        return results
