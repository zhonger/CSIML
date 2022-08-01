# import unittest
import json

import numpy as np
import pandas as pd
from sklearn import preprocessing

import SIML.CV.cv as cv
import SIML.Plot.plot as plot
from SIML.Method.base import PredictOnlyMajor, PredictWithMinor, SIML
from SIML.Analytics.analysis import AnalysisResults, AnalysisSummarys, SaveMetrics
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

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
        n_list = json.loda(fp)
    return n_list

if __name__ == '__main__':
    # data = pd.read_excel("bandgap.xlsx", sheet_name=2).iloc[:, [0, -1]]
    # data = pd.read_excel("bandgap.xlsx", sheet_name="468")
    data = pd.read_excel("bandgap.xlsx", sheet_name="3895")
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

    # cv_method = "siml"
    cv_method = "pls"
    # cv_method = "jpcl"
    # cv_method = "basis1"
    # basic_model = "SVR"
    parameters = dict(C = 10, gamma = 0.01, epsilon = 0.2, eta = 1, tolerance = 0.05, delta = 0.5, nonC = False)
    # parameters = dict(C = 1033, gamma = 0.01, epsilon = 0.2, eta = 1, tolerance = 0.05, delta = 0.5)
    # cv.CalSize(data, cv_method)
    # sampling_method = "oversampling"
    # sampling_method = "undersampling"
    sampling_method = "None"

    # plot.PlotBar(data, 0.3, 0.5)
    # plot.PlotPeriod(data)
    # plot.PlotCirle(data)


    # data, X, y = MakeFeatures(data, features, stats)
    # excel_writer = pd.ExcelWriter("bandgap.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace")
    # pd.DataFrame(data).to_excel(excel_writer, sheet_name="468")
    # excel_writer.save()

    X = data.iloc[:, 3:].fillna(0).values
    X = preprocessing.MinMaxScaler().fit_transform(pd.DataFrame(X))  # Normalization
    y = data.iloc[:, 1].values

    cv.CalSize(data, cv_method, sampling_method=sampling_method)
    split_data = cv.SplitData(data, cv_method, sampling_method=sampling_method)

    # results = PredictOnlyMajor(data, X, y, split_data, parameters)
    # results = PredictWithMinor(data, X, y, split_data, parameters)
    results = SIML(data, X, y, split_data, parameters)
    # results = SIML(data, X, y, split_data, parameters, "error_descend")
    # results = SIML(data, X, y, split_data, parameters, "bandgap_ascend")
    # results = SIML(data, X, y, split_data, parameters, "bandgap_descend")
    # df = px.data.gapminder()
    # app.run_server(debug=True)

    # summarys = AnalysisResults(results, "basis1")
    # summarys = AnalysisResults(results, "basis2")
    summarys = AnalysisResults(results, "siml")
    metrics = AnalysisSummarys(summarys)
    # SaveMetrics(metrics, "summary_basis1_0.2_3895.xlsx")
    # SaveMetrics(metrics, "summary_basis2_pls_0.2_3895.xlsx")
    # SaveMetrics(metrics, "summary_basis2_pls_oversampling_0.2_3895.xlsx")
    # SaveMetrics(metrics, "summary_basis2_pls_undersampling_0.2_3895.xlsx")
    # SaveMetrics(metrics, "summary_basis2_siml_oversampling_0.2_1033.xlsx")
    # SaveMetrics(metrics, "summary_basis2_siml_undersampling_0.2.xlsx")
    SaveMetrics(metrics, "summary_siml_pls_0.2_3895.xlsx")
    # SaveMetrics(metrics, "summary_siml_siml_0.2_nonC.xlsx")
    # SaveMetrics(metrics, "summary_siml_siml_0.2_error_descend.xlsx")
    # SaveMetrics(metrics, "summary_siml_siml_0.2_bandgap_ascend.xlsx")
    # SaveMetrics(metrics, "summary_siml_siml_0.2_bandgap_descend.xlsx")


    # a_list = np.array(results).tolist()
    # WriteList(a_list)

    plot.PlotPrediction(results[2])
    # plot.PlotPredictionMAPE(results[2])
    # plot.OldPlotPredictionMAPE(results[2])
    # plot.PlotPredictionError(results[2])
    # plot.OldPlotPredictionError(results[2])
    # fig = px.scatter(x=results[0][1], y=results[0][2])
    # fig.show()

    # RealResults(data, results)
    print("OK")
