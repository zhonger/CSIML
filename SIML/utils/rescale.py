"""Rescale helper module"""

import copy
import math
import os
import pickle
import random
from dataclasses import dataclass

import pandas as pd

from SIML.analysis.main import Analysis
from SIML.model.iml import IML
from SIML.model.basis2 import Basis2Model


@dataclass
class ScaleInfo:
    seed: int
    path: str
    sheet_name: str
    ratio: float
    data: pd.DataFrame
    maj_set: list
    min_set: list


@dataclass
class PerfInfo:
    random_state: int
    n_jobs: int
    parameters: dict
    mpi_mode: bool
    ranges: list
    multiples: list


def read_data(**kws) -> ScaleInfo:
    """Read original dataset from excel file

    Args:


    Returns:
        ScaleInfo: _description_
    """
    seed = kws.get("seed", 10)
    path = kws.get("path", "bandgap.xlsx")
    sheet_name = kws.get("sheet_name", "3895")
    data = pd.read_excel(path, sheet_name)
    threshold = kws.get("threshold", 5)
    maj_set = data[data["Experimental"] < threshold].index.values.tolist()
    min_set = data[data["Experimental"] >= threshold].index.values.tolist()
    ratio = kws.get("ratio", len(maj_set) / (len(maj_set) + len(min_set)))
    return ScaleInfo(seed, path, sheet_name, ratio, data, maj_set, min_set)


def generate_dataset(total, **kws):
    info = read_data(**kws)
    random.seed(info.seed)
    if info.ratio > 1:
        raise ValueError(f"ratio should be smaller than 1 (now is {info.ratio})")
    maj_num = math.ceil(total * info.ratio)
    min_num = total - maj_num
    maj_set_s = random.sample(info.maj_set, maj_num)
    min_set_s = random.sample(info.min_set, min_num)
    set_s = copy.deepcopy(maj_set_s)
    set_s.extend(min_set_s)
    data_s = info.data.iloc[set_s, :].reset_index(drop=True)
    return data_s


def perf_init(**kws):
    random_state = kws.get("random_state", 10)
    n_jobs = kws.get("n_jobs", 25)
    parameters = kws.get(
        "parameters",
        {
            "C": 10,
            "gamma": 0.01,
            "epsilon": 0.2,
        },
    )
    ranges = kws.get("ranges", [5, 4, 3, 2, 1])
    multiples = kws.get("multiples", [5, 4, 3, 2, 1])
    mpi_mode = kws.get("mpi_mode", False)
    return PerfInfo(random_state, n_jobs, parameters, mpi_mode, ranges, multiples)


def perf(data: pd.DataFrame, tasks: list, path: str, **kws):
    dataset = data.shape[0]
    info = perf_init(**kws)
    ts = []
    for task in tasks:
        if task[0] == "basis2":
            model = Basis2Model(
                data,
                cv_method=task[1],
                sampling_method=task[2],
                random_state=info.random_state,
                n_jobs=info.n_jobs,
                mpi_mode=info.mpi_mode,
                parameters=info.parameters,
            )
        elif task[0] == "siml":
            C_limit = kws.get("C_limit", 0.51)
            model = IML(
                data,
                cv_method=task[1],
                random_state=info.random_state,
                ranges=info.ranges,
                multiples=info.multiples,
                n_jobs=info.n_jobs,
                mpi_mode=info.mpi_mode,
                parameters=info.parameters,
                C_limit=C_limit,
                opt_C=task[3],
            )
        print(model)
        results = model.fit_predict()
        if task[2] == "oversampling" or task[2] == "undersampling":
            method = task[2]
            filename = f"{dataset}_{task[0]}_{task[2]}"
        elif task[0] == "siml":
            if task[3]:
                method = "SIML with C optimization"
                filename = f"{dataset}_{task[0]}_C"
            else:
                method = "SIML without C optimization"
                filename = f"{dataset}_{task[0]}_nonC"
        else:
            method = task[0]
            filename = f"{dataset}_{task[0]}"
        a1 = Analysis(results=results, method=method)
        ts.append(a1)
        filename = f"{path}/{filename}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(a1, f)
        print(a1)
    return ts


def rescale(total: int, **kws) -> list:
    """Rescale dataset with specified total number and others

    Args:
        total (int): the total number of dataset.
        seed (int, optional): the random seed for dataset rescaling. Defaults to 10.


    Returns:
        list: analysis result objects.
    """
    info = read_data(**kws)
    data = generate_dataset(total, **kws)
    seed = kws.get("seed", 10)
    print(
        f"Total: {total}, sheet_name: {info.sheet_name}, "
        f"seed: {seed}, ratio: {info.ratio:.3f}\n"
    )
    path = f"tmp/rescale/{info.sheet_name}_{info.ratio/(1-info.ratio):.2f}"
    if not os.path.exists(path):
        os.makedirs(path)
    tasks = kws.get("tasks", [["siml", "siml", None, False]])
    if "tasks" in kws:
        del kws["tasks"]
    ts = perf(data, tasks, path, **kws)
    return ts
