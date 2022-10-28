import os
import sys, time

sys.path.append('/home/lisz/work/SIML/')

import numpy as np
import pandas as pd
from sklearn import preprocessing

from SIML.CV import cv
from SIML.Method.base import SIML
from SIML.Analytics.analysis import LoadResults, AnalysisCosts


if __name__ == '__main__':
    
    path = "/home/lisz/work/SIML/test/"
    os.chdir(path)
    
    # sheet_names = dict(EXP="468", LDA="468-LDA", PBE="468-PBE", SCAN="468-SCAN", HSE06 = "468-HSE06", HSE06mix = "468-HSE06_mix")
    sheet_names = dict(EXP="3895")
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
            # C_list = [10e-1]
            C = 10
            # data_label = "468"
            data_label = "3895"
            random_state = 50 # 3, 10, 30, 50
                            
            parameters = dict(C=C, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.5, parallel=True, n_jobs=-1, wMAE=False, tolerance2=0.05, method=method, cv_method=cv_method, sampling_method=sampling_method, data_sheet=data_sheet, random_state=random_state, wb=True)
            
            # Split data  
            split_data = cv.SplitData(data, cv_method, sampling_method=sampling_method, random_state=random_state)
                
            # Prepare the path and filename
            if sampling_method == "None":
                filename = method + "_" + cv_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet
            else:
                filename = method + "_" + cv_method + "_" + sampling_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet

            # Main
            start_at = time.time()
            results = SIML(data, X, y, split_data, parameters)
            end_at = time.time()
            print('%s used %s s' % (filename, (end_at - start_at)))  
    
            # filename = "468/siml_op/siml_op_0.2_10_EXP_results.xlsx"
            filename = "random_seed/" + str(random_state) + "/" + data_label + "/siml_op_" + job[3] + "_" + job[4] +"/siml_op_0.2_10_EXP_results.xlsx"
            siml_results = LoadResults(filename, True)
            
            filename = "random_seed/" + str(random_state) + "/" + data_label + "/siml_op_" + job[3] + "_" + job[4] +"/siml_op_" + job[3] + "_" + job[4] +"_costs.xlsx"
            AnalysisCosts(siml_results, results, filename)
    
    print("OK")