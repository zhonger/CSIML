# import unittest
import os
import sys, time

sys.path.append('/home/lisz/work/SIML/')
import json

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from sklearn import preprocessing

import SIML.CV.cv as cv
import SIML.Plot.plot as plot
from SIML.Analytics.analysis import (AnalysisResults, AnalysisSummarys,
                                     SaveMetrics, SaveResults)
from SIML.Method.base import SIML, PredictOnlyMajor, PredictWithMinor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Slider(1, 625, 1, value=3, marks=None, id="my-slider",
               tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(id='slider-output-container')
])

@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return 'You have selected "{}"'.format(value)

def WriteList(a_list):
    with open("results.json", "w") as fp:
        json.dump(a_list, fp)

def ReadList():
    with open("results.json", "rb") as fp:
        n_list = json.load(fp)
    return n_list

if __name__ == '__main__':
    
    # sheet_names = dict(EXP="468", LDA="468-LDA", PBE="468-PBE", SCAN="468-SCAN", HSE06 = "468-HSE06", HSE06mix = "468-HSE06_mix")
    sheet_names = dict(EXP="3895")
    # data_sheets = ["EXP", "LDA", "PBE", "SCAN", "HSE06", "HSE06mix"]
    # data_sheets = ["PBE", "SCAN", "HSE06", "HSE06mix"]
    # data_sheets = ["SCAN", "HSE06", "HSE06mix"]
    # data_sheets = ["LDA"]
    data_sheets = ["EXP"]
    # data_sheet  = "EXP"

    # data = pd.read_excel("bandgap.xlsx", sheet_name=2).iloc[:, [0, -1]]
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468-LDA")
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468-PBE")
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468-SCAN")
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468-HSE06")
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468-HSE06_mix")
    # data = pd.read_excel("bandgap.xlsx", sheet_name="3895")
    
    features = ["Number", "AtomicWeight", "Period", "Group", "Family",
                "LQuantumNumber", "MendeleevNumber",
                "AtomicRadius", "CovalentRadius", "ZungerRadius", "IonicRadius", "CrystalRadius",
                "Electronegativity", "MartynovBatsanovEN", "GordyEN", "MullikenEN", "AllenEN",
                "MetallicValence",
                "NValence", "NsValence", "NpValence", "NdValence", "NUnfilled",
                "FirstIonizationEnergy", "Polarizability",
                "MeltingT", "BoilingT", "Density",
                "SpecificHeat", "HeatFusion", "HeatVaporization", "ThermalConductivity", "HeatAtomization",
                "CohesiveEnergy"]
    stats = ["maximum", "minimum", "avg_dev", "mean"]

    # Jobs
    jobs = [
            # ["basis1", "basis1", "None"],
            ["basis2", "pls", "None"],
            # ["basis2", "pls", "oversampling"],
            # ["basis2", "pls", "undersampling"],
            # ["siml", "pls", "None", "O1", "C3"],
            # ["basis2", "jpcl", "None"],
            # ["basis2", "jpcl", "oversampling"],
            # ["basis2", "jpcl", "undersampling"],
            # ["siml", "jpcl", "None"],
            # ["basis2", "siml", "None"],
            # ["basis2", "siml", "oversampling"],
            # ["basis2", "siml", "undersampling"],
            # ["siml", "siml", "None"]
            ]
    
    # Run tasks
    for data_sheet in data_sheets:
        path = "/home/lisz/work/SIML/test/"
        os.chdir(path)
        # path = os.getcwd() + "/"
        print("Now we are in %s" % path)
        
        data = pd.read_excel("bandgap.xlsx", sheet_name=sheet_names[data_sheet])
        print("Find dataset file")
        
        for job in jobs:
            method = job[0]
            cv_method = job[1]
            sampling_method = job[2]
        
            # basic_model = "SVR"
            # parameters = dict(C = 10, gamma = 0.01, epsilon = 0.2, eta = 1, tolerance = 0.005, delta = 0.5, nonC = False, parallel=True, n_jobs=50) # With C optimization
            # parameters = dict(C=10, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.5, nonC=False, parallel=False, n_jobs=50, nonOrder=True) # With C optimization
            # parameters = dict(C=10, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.1, nonC=False, parallel=False, n_jobs=50, nonOrder=True) # With C optimization
            # parameters = dict(C=10, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.5, nonC=False, parallel=True, n_jobs=50, nonOrder=False) # 
            parameters = dict(C=10, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.5, parallel=True, n_jobs=-1, wMAE=False, tolerance2=0.05) # 
            # parameters = dict(C=10, gamma=0.01, epsilon=0.2, eta=1, tolerance=0.005, delta=0.5, nonC=True, parallel=True, n_jobs=50, nonOrder=False) # 
            # parameters = dict(C = 10, gamma = 0.01, epsilon = 0.2, eta = 1, tolerance = 0.005, delta = 0.5, nonC = True) # Without  C optimization
            # parameters = dict(C = 1000, gamma = 0.01, epsilon = 0.2, eta = 1, tolerance = 0.005, delta = 0.5)
            # cv.CalSize(data, cv_method)
            
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

            # plot.PlotBar(data, 1, 5)
            # plot.PlotPeriod(data)
            # plot.PlotCirle(data)

            # data, X, y = MakeFeatures(data, features, stats)
            # excel_writer = pd.ExcelWriter("bandgap.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace")
            # pd.DataFrame(data).to_excel(excel_writer, sheet_name="468")
            # excel_writer.save()


            # Data preprocessing
            data.sort_values(by="Experimental", inplace=True, ascending=True)
            X = data.iloc[:, 3:].fillna(0).values
            X = preprocessing.MinMaxScaler().fit_transform(pd.DataFrame(X))  # Normalization
            y = data.iloc[:, 1].values
            
            # Prepare the path and filename
            if method == "siml":
                if sampling_method == "None":
                    filename = method + "_" + cv_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet + "_" + str(parameters["delta"]) + "_" + op_method + "_" + C_op_method
                else:
                    filename = method + "_" + cv_method + "_" + sampling_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet + "_" + str(parameters["delta"]) + "_" + op_method + "_" + C_op_method
            else:
                if sampling_method == "None":
                    filename = method + "_" + cv_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet
                else:
                    filename = method + "_" + cv_method + "_" + sampling_method + "_" + str(parameters["epsilon"]) + "_" + str(parameters["C"]) + "_" + data_sheet
    
            directory = path + filename
            if os.path.exists(directory):
                os.chdir(directory)
            else:
                os.mkdir(directory)
                os.chdir(directory)


            # cv.CalSize(data, cv_method, sampling_method=sampling_method)
            # Split data
            split_data = cv.SplitData(data, cv_method, sampling_method=sampling_method)

            # Main
            start_at = time.time()
            if method == "basis1":
                results = PredictOnlyMajor(data, X, y, split_data, parameters)
            elif method == "basis2":
                results = PredictWithMinor(data, X, y, split_data, parameters)
            else:
                results = SIML(data, X, y, split_data, parameters)
            end_at = time.time()
            print('%s used %s s' % (filename, (end_at - start_at)))


            # Analysis results
            summarys = AnalysisResults(results, method)
            metrics = AnalysisSummarys(summarys)
            SaveMetrics(metrics, "summary_" + filename + ".xlsx")                               
            SaveResults(results, "summary_" + filename + ".xlsx")


            # Plot results
            if cv_method == "siml":
                figname = filename + "_2"
                plot.PlotPrediction(results[2], method, figname)
                print(figname + ": OK")
            else:
                for i in range(len(split_data)):
                    figname = filename + "_"+ str(i)
                    plot.PlotPrediction(results[i], method, figname)
                    print(figname + ": OK")

                    # plot.PlotPredictionMAPE(results[2])
                    # plot.OldPlotPredictionMAPE(results[2])
                    # plot.PlotPredictionError(results[2])
                    # plot.OldPlotPredictionError(results[2])
                    # fig = px.scatter(x=results[0][1], y=results[0][2])
                    # fig.show()
