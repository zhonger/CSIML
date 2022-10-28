import os
import sys, time

sys.path.append("/home/lisz/work/SIML/")

import numpy as np
import scipy
import physbo
import itertools
import pandas as pd
from sklearn import preprocessing

import SIML.CV.cv as cv
import SIML.Plot.plot as plot
from SIML.Analytics.analysis import (
    AnalysisResults,
    AnalysisSummarys,
    AnalysisHyperparameter,
    SaveMetrics,
    SaveResult,
    SaveResults,
    SaveHyperparameterMetrics,
    LoadResults,
    LoadSummarys,
)
from SIML.Method.base import SIML, PredictOnlyMajor, PredictWithMinor


if __name__ == "__main__":

    path = "/home/lisz/work/SIML/test/"
    os.chdir(path)

    sheet_names = dict(
        EXP="468",
        LDA="468-LDA",
        PBE="468-PBE",
        SCAN="468-SCAN",
        HSE06="468-HSE06",
        HSE06mix="468-HSE06_mix",
    )
    # sheet_names = dict(EXP="3895")
    data_sheets = ["EXP"]

    jobs = [
        # ["basis1", "basis1", "None"],
        # ["basis2", "op", "None"],
        # ["basis2", "op", "oversampling"],
        # ["basis2", "pls", "undersampling"],
        ["siml", "op", "None", "O1", "C2"],
        # ["siml", "op", "None", "O2", "C2"],
        # ["siml", "op", "None", "O3", "C2"],
        # ["siml", "op", "None", "O4", "C2"],
        # ["basis2", "jpcl", "None"],
        # ["basis2", "jpcl", "oversampling"],
        # ["basis2", "jpcl", "undersampling"],
        # ["siml", "jpcl", "None"],
        # ["basis2", "siml", "None"],
        # ["basis2", "siml", "oversampling"],
        # ["basis2", "siml", "undersampling"],
        # ["siml", "siml", "None"]
    ]

    for data_sheet in data_sheets:
        data = pd.read_excel("bandgap.xlsx", sheet_name=sheet_names[data_sheet])

        # Data preprocessing
        data.sort_values(by="Experimental", inplace=True, ascending=True)
        X = data.iloc[:, 3:].fillna(0).values
        X = preprocessing.MinMaxScaler().fit_transform(pd.DataFrame(X))  # Normalization
        y = data.iloc[:, 1].values

        for job in jobs:
            method = job[0]
            cv_method = job[1]
            sampling_method = job[2]
            C_list = [10]
            # C_list = [10e-1, 10, 20, 30, 40, 0.5*10e1, 0.6*10e1, 0.7*10e1, 0.8*10e1, 0.9*10e1, 10e1, 1.1*10e1, 1.2*10e1, 1.3*10e1, 1.4*10e1, 1.5*10e1, 1.6*10e1, 1.7*10e1, 1.8*10e1, 1.9*10e1, 0.2*10e2, 0.3*10e2, 0.4*10e2, 0.5*10e2, 0.6*10e2, 0.7*10e2, 0.8*10e2, 0.9*10e2, 10e2] # for begin test
            # C_list = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10, 20, 30, 40, 0.5*10e1, 0.6*10e1, 0.7*10e1, 0.8*10e1, 0.9*10e1, 10e1, 1.1*10e1, 1.2*10e1, 1.3*10e1, 1.4*10e1, 1.5*10e1, 1.6*10e1, 1.7*10e1, 1.8*10e1, 1.9*10e1, 0.2*10e2, 0.3*10e2, 0.4*10e2, 0.5*10e2, 0.6*10e2, 0.7*10e2, 0.8*10e2, 0.9*10e2, 10e2, 10e3, 10e4, 10e5] # for basis2 or oversampling
            # C_list = [10e-1, 10, 0.5*10e1, 0.6*10e1, 0.7*10e1, 0.8*10e1, 0.9*10e1, 10e1, 1.1*10e1, 1.2*10e1, 1.3*10e1, 1.4*10e1, 1.5*10e1, 1.6*10e1, 1.7*10e1, 1.8*10e1, 1.9*10e1, 0.2*10e2, 0.3*10e2, 0.4*10e2, 0.5*10e2, 0.6*10e2, 0.7*10e2, 0.8*10e2, 0.9*10e2, 10e2, 0.5*10e3] # for siml(468)
            # C_list = [10e-1, 10, 0.5*10e1, 0.6*10e1, 0.7*10e1, 0.8*10e1, 0.9*10e1, 10e1, 1.1*10e1, 1.2*10e1, 1.3*10e1, 1.4*10e1, 1.5*10e1, 1.6*10e1, 1.7*10e1, 1.8*10e1, 1.9*10e1, 0.2*10e2, 0.3*10e2, 0.4*10e2, 0.5*10e2, 0.6*10e2, 0.7*10e2, 0.8*10e2, 0.9*10e2, 10e2] # for all 3895

            # R1
            # ranges = [5, 4, 3, 2, 1]
            # multiples = [5, 4, 3, 2, 1]

            # R2
            # ranges = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]
            # multiples = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]

            # R3
            # ranges = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]
            # multiples = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

            # R4
            # ranges = [5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            # multiples = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2]
            ranges_m = [
                [5, 4, 3, 2, 1],
                [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1],
                [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1],
                [5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ]
            multiples_m = [
                [5, 4, 3, 2, 1],
                [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2],
            ]

            # Split data
            data_label = "468"
            # data_label = "3895"
            random_state = 10  # 3, 10, 30, 50
            split_data = cv.SplitData(
                data,
                cv_method,
                sampling_method=sampling_method,
                random_state=random_state,
            )

            # Prepare the path and filename
            if sampling_method == "None":
                directory = "%srandom_seed/%s/%s/%s_%s_%s_%s" % (
                    path,
                    random_state,
                    data_label,
                    method,
                    cv_method,
                    job[3],
                    job[4],
                )
            else:
                directory = "%srandom_seed/%s/%s/%s_%s_%s" % (
                    path,
                    random_state,
                    data_label,
                    method,
                    cv_method,
                    sampling_method,
                )

            if os.path.exists(directory):
                os.chdir(directory)
            else:
                os.makedirs(directory)
                os.chdir(directory)

            op_results = []

            # Check C
            for C in C_list:
                for i in range(len(ranges_m)):
                    ranges = ranges_m[i]
                    multiples = multiples_m[i]
                    if C >= 1:
                        C = round(C, 1)
                    parameters = dict(
                        C=C,
                        gamma=0.01,
                        epsilon=0.2,
                        eta=1,
                        tolerance=0.005,
                        delta=0.5,
                        parallel=False,
                        n_jobs=-1,
                        wMAE=False,
                        tolerance2=0.05,
                        method=method,
                        cv_method=cv_method,
                        sampling_method=sampling_method,
                        data_sheet=data_sheet,
                        random_seed=random_state,
                        wb=False,
                        log=True,
                        phar=False,
                        rop=True,
                        ranges=ranges,
                        multiples=multiples,
                    )
                    # parameters = dict(C=C, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.015, delta=0.5, parallel=False, n_jobs=-1, wMAE=False, tolerance2=0.05, method=method, cv_method=cv_method, sampling_method=sampling_method, data_sheet=data_sheet, random_seed = random_state, wb=False, log=True, phar=False)

                    # Add parameters for SIML method
                    if method == "siml":
                        if len(job) <= 3:
                            op_method = "O1"
                            C_op_method = "C1"
                        elif len(job) == 4:
                            op_method = job[3]
                            C_op_method = "C1"
                        elif len(job) == 5:
                            op_method = job[3]
                            C_op_method = job[4]

                        parameters["op_method"] = op_method
                        match C_op_method:
                            case "C1":
                                parameters["nonC"] = False
                                parameters["nonOrder"] = False
                            case "C2":
                                parameters["nonC"] = True
                                parameters["nonOrder"] = False
                            case "C3":
                                parameters["nonC"] = False
                                parameters["nonOrder"] = True

                    # Prepare the path and filename
                    if sampling_method == "None":
                        filename = "%s_%s_%s_%s_%s" % (
                            method,
                            cv_method,
                            parameters["epsilon"],
                            parameters["C"],
                            data_sheet,
                        )
                    else:
                        filename = "%s_%s_%s_%s_%s_%s" % (
                            method,
                            cv_method,
                            sampling_method,
                            parameters["epsilon"],
                            parameters["C"],
                            data_sheet,
                        )

                    # Main
                    start_at = time.time()
                    if method == "basis1":
                        results = PredictOnlyMajor(data, X, y, split_data, parameters)
                    elif method == "basis2":
                        results = PredictWithMinor(data, X, y, split_data, parameters)
                    else:
                        results = SIML(data, X, y, split_data, parameters)
                    end_at = time.time()
                    print("%s used %s s" % (filename, (end_at - start_at)))

                    # Load Results from excels
                    # filename = filename + "_results.xlsx"
                    # results = LoadResults(filename)  # without parallel
                    # results = LoadResults(filename, True)  # with parallel (10 cores)

                    op_results.append(results)

            metrics_filename = method + "_" + cv_method + "_metrics.xlsx"
            op_metrics = SaveHyperparameterMetrics(
                op_results, filename=metrics_filename
            )  # write results and metrics to excel
            # op_metrics =  SaveHyperparameterMetrics(op_results, filename=metrics_filename, save_result=False) # only write metrics to excel
            op_summarys = AnalysisHyperparameter(op_results)
            # op_summarys = LoadSummarys(metrics_filename)

            if sampling_method == "None":
                result_filename = method + "_" + cv_method
            else:
                result_filename = method + "_" + cv_method + "_" + sampling_method
            plot_filename = result_filename + "_" + data_sheet
            plot.PlotHyperparametr(op_summarys, plot_filename)
            plot.PlotHyperparametr(op_summarys, plot_filename, False)
            plot.PlotHyperparametr(op_summarys, plot_filename, metric="rmse")
