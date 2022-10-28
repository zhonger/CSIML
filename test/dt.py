import os
import sys, time

sys.path.append('/home/lisz/work/SIML/')

import numpy as np
import scipy
import physbo
import itertools
import pandas as pd
from sklearn import preprocessing

import SIML.CV.cv as cv
import SIML.Plot.plot as plot
from SIML.Analytics.analysis import (AnalysisResults, AnalysisSummarys, AnalysisHyperparameter,
                                     SaveMetrics, SaveResult, SaveResults, SaveHyperparameterMetrics,
                                     LoadResults, LoadSummarys)
from SIML.Method.base import SIML, PredictOnlyMajor, PredictWithMinor

if __name__ == '__main__':
    
    path = "/home/lisz/work/SIML/test/"
    os.chdir(path)

    random_seeds = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    data_labels = [468, 3895]
    # random_seeds = [3, 10, 30, 50]
    # data_labels = [468]
    C = 10
    
    selected_columns = ["random_seed", "C", "data_label",
                        "test_mae", "test_majority_mae", "test_minority_mae", "test_mae_difference", 
                        "test_rmse", "test_majority_rmse", "test_minority_rmse", "test_rmse_difference"]
    # methods = ["basis2_op", "basis2_op_oversampling", "siml_op_O1_C2"]
    # metrics_filenames = ["basis2_op_metrics.xlsx", "basis2_op_metrics.xlsx", "siml_op_metrics.xlsx"]
    # methods = ["siml_op_O2_C2", "siml_op_O3_C2", "siml_op_O4_C2"]
    # metrics_filenames = ["siml_op_metrics.xlsx", "siml_op_metrics.xlsx", "siml_op_metrics.xlsx"]
    methods = ["siml_op_O1_C2"]
    metrics_filenames = ["siml_op_metrics.xlsx"]
    
    for i in range(len(methods)):
        method = methods[i]
        metrics_filename = metrics_filenames[i]
        metrics = []
        
        for random_seed in random_seeds:
            for data_label in data_labels:
                filename = path + "random_seed/" + str(random_seed) + "/" + str(data_label) +"/"+ method +"/" + metrics_filename
                data = pd.read_excel(filename, sheet_name="op_metrics", index_col=[0])
                metric = data[data["C"]==C]
                metric.insert(0, "data_label", data_label)
                metric.insert(0, "random_seed", random_seed)
                metric = metric.mean()
                metrics.append(metric)
                
        metrics = pd.DataFrame(metrics)
        metrics["test_mae_difference"] = metrics["test_minority_mae"] / metrics["test_majority_mae"]
        metrics["test_rmse_difference"] = metrics["test_minority_rmse"] / metrics["test_majority_rmse"]
        
        results = metrics.loc[:, selected_columns]
        
        results_split = []
        for i in range(len(data_labels)):
            results_split.append(results[results["data_label"]==data_labels[i]])
            
        save_filename = "random_seeds_" + method + ".xlsx"
        with pd.ExcelWriter(save_filename) as writer:
            results.to_excel(excel_writer=writer, sheet_name="summary", index=False)
            for i in range(len(data_labels)):
                results_split[i].to_excel(excel_writer=writer, sheet_name=str(data_labels[i]), index=False)
        
        print("%s finished." % method)