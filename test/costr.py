import os
import sys, time

sys.path.append("/home/lisz/work/SIML/")

import numpy as np
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

    # sheet_names = dict(
    #     EXP="468",
    #     LDA="468-LDA",
    #     PBE="468-PBE",
    #     SCAN="468-SCAN",
    #     HSE06="468-HSE06",
    #     HSE06mix="468-HSE06_mix",
    # )
    sheet_names = dict(EXP="3895")
    data_sheets = ["EXP"]

    jobs = [
        ["siml", "op", "None", "O1", "C2"],
        # ["siml", "op", "None", "O2", "C2"],
        # ["siml", "op", "None", "O3", "C2"],
        # ["siml", "op", "None", "O4", "C2"],
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
            # data_label = "468"
            data_label = "3895"
            random_state = 10  # 3, 10, 30, 50
            split_data = cv.SplitData(
                data,
                cv_method,
                sampling_method=sampling_method,
                random_state=random_state,
            )

            op_results = []

            # Check C
            for C in C_list:
                if C >= 1:
                    C = round(C, 1)
                for i in range(len(ranges_m)):
                    ranges = ranges_m[i]
                    multiples = multiples_m[i]

                    # Prepare the path and filename
                    # if sampling_method == "None":
                    #     directory = "%srandom_seed/%s/%s/%s_%s_%s_%s_R%s" % (
                    #         path,
                    #         random_state,
                    #         data_label,
                    #         method,
                    #         cv_method,
                    #         job[3],
                    #         job[4],
                    #         i + 1,
                    #     )
                    # else:
                    #     directory = "%srandom_seed/%s/%s/%s_%s_%s" % (
                    #         path,
                    #         random_state,
                    #         data_label,
                    #         method,
                    #         cv_method,
                    #         sampling_method,
                    #     )

                    # if os.path.exists(directory):
                    #     os.chdir(directory)
                    # else:
                    #     os.makedirs(directory)
                    #     os.chdir(directory)

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
                        costr=True,
                        ranges=ranges,
                        multiples=multiples,
                    )

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
                    print("%s_R%s used %s \n" % (filename, i + 1, (end_at - start_at)))
                    op_results.append(results)

            for i in range(len(ranges_m)):
                filename = "/home/lisz/work/SIML/test/costr_%s.xlsx" % (data_label)
                sheet_name = "R%s" % (i + 1)
                op_result = op_results[i][0]
                header = ["No.", "step", "cost"]
                if i == 0:
                    with pd.ExcelWriter(path=filename) as writer:
                        pd.DataFrame(op_result).to_excel(
                            excel_writer=writer,
                            sheet_name=sheet_name,
                            index=None,
                            header=header,
                        )
                else:
                    with pd.ExcelWriter(
                        path=filename, mode="a", if_sheet_exists="replace"
                    ) as writer:
                        pd.DataFrame(op_result).to_excel(
                            excel_writer=writer,
                            sheet_name=sheet_name,
                            index=None,
                            header=header,
                        )
