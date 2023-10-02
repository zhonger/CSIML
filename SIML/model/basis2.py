"""The module for Basis2 model"""
import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpi4py import MPI
from tqdm import tqdm

if os.getenv("INTELEX", "False") == "True":
    from sklearnex import patch_sklearn

    patch_sklearn()


from SIML.analysis.analysis import index_to_name
from SIML.utils.timer import timer

from ._base import BaseModel


class Basis2Model(BaseModel):
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
        regr = self.regr
        train = np.append(df[0], df[3])
        X_train = self.X[train, :]
        y_train = self.y[train]
        regr = regr.fit(X_train, y_train)
        return regr

    def fit(self) -> list:
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
            regrs = Parallel(self.n_jobs, verbose=3)(
                delayed(self.__fit__)(splited_data[i])
                for i in tqdm(range(self.iterations))
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
            results = Parallel(self.n_jobs, verbose=3)(
                delayed(self.__predict__)(splited_data[i], regrs[i])
                for i in tqdm(range(self.iterations))
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
        regrs = self.fit()
        results = self.predict(regrs)
        return results
