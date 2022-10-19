import pprint
import sys
from collections import Counter
from multiprocessing import Pool
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import SIML.CV.cv as cv
from SIML.Analytics.analysis import RealResults
from tqdm import tqdm


def Evaluation(y: np.array, y_pred: np.array):
    error = abs(y - y_pred)
    mae = mean_absolute_error(y, y_pred)
    var = median_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    std = np.std(abs(y - y_pred))

    return error, mae, var, rmse, mape, r2, std


def PredictOnlyMajor(data: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, split_data: pd.DataFrame, parameters: dict,
                     basic_model: str = "SVR"):
    """

    :param X:
    :param y:
    :param split_data:
    :return:
    """

    if basic_model == "SVR":
        regr = SVR(kernel="rbf", C=parameters["C"], gamma=parameters["gamma"])
    if basic_model == "DT":
        regr = DecisionTreeRegressor(max_depth=10)
    if basic_model == "RF":
        regr = RandomForestRegressor(n_estimators=10)

    iterations = split_data.__len__()
    results = []

    pbar = tqdm(total=iterations)
    pbar.set_description("Basis 1")
    for i in range(iterations):
        result = []
        train = split_data[i][0]
        validation = split_data[i][1]
        majority_test = split_data[i][2]
        minority_test = split_data[i][3]
        majority_test_num = majority_test.__len__()
        test = np.append(majority_test, minority_test)

        X_train = X[train, :]
        y_train = y[train]
        X_validation = X[validation, :]
        y_validation = y[validation]
        X_test = X[test, :]
        y_test = y[test]
        y_majority_test = y_test[:majority_test_num]
        y_minority_test = y_test[majority_test_num:]

        regr = regr.fit(X_train, y_train)
        y_train_pred = regr.predict(X_train)
        y_validation_pred = regr.predict(X_validation)
        y_test_pred = regr.predict(X_test)
        y_majority_test_pred = y_test_pred[:majority_test_num]
        y_minority_test_pred = y_test_pred[majority_test_num:]

        # Update index to materials name

        train = RealResults(data, train)
        validation = RealResults(data, validation)
        majority_test = RealResults(data, majority_test)
        minority_test = RealResults(data, minority_test)

        result.append(train)
        result.append(y_train)
        result.append(y_train_pred)
        result.append(validation)
        result.append(y_validation)
        result.append(y_validation_pred)
        result.append(majority_test)
        result.append(y_majority_test)
        result.append(y_majority_test_pred)
        result.append(minority_test)
        result.append(y_minority_test)
        result.append(y_minority_test_pred)
        results.append(result)
        pbar.update(1)

    return results


def Predict(data: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, df: pd.DataFrame, parameters: dict, basic_model: str = "SVR"):
    result = []
    train = np.append(df[0], df[3])
    validation = np.append(df[1], df[4])
    test = np.append(df[2], df[5])
    X_train = X[train, :]
    y_train = y[train]
    X_validation = X[validation, :]
    y_validation = y[validation]
    X_test = X[test, :]
    y_test = y[test]

    if basic_model == "SVR":
        regr = SVR(kernel="rbf", C=parameters["C"], gamma=parameters["gamma"], epsilon=parameters["epsilon"])
    if basic_model == "DT":
        regr = DecisionTreeRegressor(max_depth=10)
    if basic_model == "RF":
        regr = RandomForestRegressor(n_estimators=10)

    regr = regr.fit(X_train, y_train)
    y_train_pred = regr.predict(X_train)
    y_validation_pred = regr.predict(X_validation)
    y_test_pred = regr.predict(X_test)

    result.append(RealResults(data, df[0]))
    result.append(y[df[0]])
    result.append(y_train_pred[:len(df[0])])

    result.append(RealResults(data, df[1]))
    result.append(y[df[1]])
    result.append(y_validation_pred[:len(df[1])])

    result.append(RealResults(data, df[2]))
    result.append(y[df[2]])
    result.append(y_test_pred[:len(df[2])])

    result.append(RealResults(data, df[3]))
    result.append(y[df[3]])
    result.append(y_train_pred[len(df[0]):])

    result.append(RealResults(data, df[4]))
    result.append(y[df[4]])
    result.append(y_validation_pred[len(df[1]):])

    result.append(RealResults(data, df[5]))
    result.append(y[df[5]])
    result.append(y_test_pred[len(df[2]):])

    counts = Counter(df[3])
    result.append(RealResults(data, list(counts.keys())))
    result.append(counts.values())
    
    result.append(list(parameters.keys()))
    result.append(list(parameters.values()))

    return result
            

def PredictWithMinor(data: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, split_data: pd.DataFrame, parameters: dict,
                     basic_model: str = "SVR"):
    iterations = split_data.__len__()
    if parameters["parallel"]:
        results = (
            Parallel(n_jobs=parameters["n_jobs"])
            (delayed(Predict)(data, X, y, split_data[i], parameters)
             for i in tqdm(range(iterations)))
        )
    else:
        
        results = []
        pbar = tqdm(total=iterations)
        pbar.set_description("Basis 2")
        for df in split_data:
            result = Predict(data, X, y, df, parameters)
            results.append(result)
            pbar.update(1)

    print("OK")

    return results


def TrainModel(regr, X_train, y_train, sample_weights):
    regr = regr.fit(X_train, y_train, sample_weights)
    y_train_pred = regr.predict(X_train)

    error, mae, variance, rmse, mape, r2, std = Evaluation(
        y_train, y_train_pred
    )

    return regr, error, mae, variance, rmse, mape, r2, std


def TestError(regr, X_test, y_test, test, method: str="O1"):
    """

    :param regr:
    :param X_test:
    :param y_test:
    :param test:
    :param method: ascend (smallest -> largest), descend (largest -> smallest) for error or bandgap
    :return:
    """
    y_pred = regr.predict(X_test)
    test_error = y_pred - y_test

    match method:
        case "O1": # error_ascend
            index = np.argmin(abs(test_error))
        case "O2": # error_descend
            index = np.argmax(abs(test_error))
        case "O3": # bandgap_ascend
            index = np.argmin(y_test)
        case "O4": # bandgap_descend
            index = np.argmax(y_test)

    return index, test[index], y_pred[index]


def LearningRate(error, min_error, step, ranges, multiples):
    m = error - min_error
    i = int(len(ranges)/2)
    t = -1
    while t == -1 :
        if m > ranges[i-1]:
            if i == 1:
                t = i
            elif m <= ranges[i-2]:
                t = i
            else:
                i = i - 1
        else:
            if i == len(ranges):
                t = i
            elif m > ranges[i]:    
                t = i - 1
            else:
                i = i + 1
        
    delta = multiples[t-1] * step
    return delta


def SIMLP(data: pd.DataFrame, X: np.array, y: np.array, split_data: list, n: int, parameters: list):
    epsilon, gamma, C, delta, eta, tolerance = (
        parameters["epsilon"],
        parameters["gamma"],
        parameters["C"],
        parameters["delta"],
        parameters["eta"],
        parameters["tolerance"]
    )
    
    if parameters["op_method"]:
        op_method = parameters["op_method"]
    else:
        op_method = "O1"
        
    majority_train_set = split_data[n - 1][0]
    majority_validation_set = split_data[n - 1][1]
    majority_test_set = split_data[n - 1][2]
    minority_train_set = split_data[n - 1][3]
    minority_validation_set = split_data[n - 1][4]
    minority_test_set = split_data[n - 1][5]

    majority_train_size = majority_train_set.shape[0]
    minority_train_size = minority_train_set.shape[0]
    majority_validation_size = majority_validation_set.shape[0]
    minority_validation_size = minority_validation_set.shape[0]
    majority_test_size = majority_test_set.shape[0]
    minority_test_size = minority_test_set.shape[0]
    # debug = self.debug

    train = majority_train_set
    validation = np.append(majority_validation_set, minority_validation_set)
    test = np.append(majority_test_set, minority_test_set)

    minority_train_num = len(minority_train_set)
    sample_weights = np.ones(len(train), dtype=float)
    coefs = np.ones(minority_train_num, dtype=float)
    order = []
    minority_train_bak = minority_train_set

    regr = SVR(kernel="rbf", epsilon=epsilon, C=C, gamma=gamma)
    
    pbar = tqdm(total=minority_train_num)
    pbar.set_description("SIML (iteration %s)" % n)

    for epoch in range(minority_train_num):
        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]

        regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(regr,
                                                                                                            X_train,
                                                                                                            y_train,
                                                                                                            sample_weights)

        best_index, best, best_pred = TestError(regr, X[minority_train_set, :], y[minority_train_set],
                                                minority_train_set, op_method)
        X_best = X[best, :]
        y_best = y[best]
        order = np.append(order, str(best))
        error = abs(y_best - best_pred)
        sample_weights = np.append(sample_weights, coefs[epoch])

        train = np.append(train, best)
        X_train = X[train, :]
        y_train = y[train]
        minority_train_set = np.delete(minority_train_set, best_index)
        limit = round(epsilon + tolerance, 6)
        ranges1 = [i * limit for i in list(range(5,0,-1))]
        multiples1 = list(range(5,0,-1))

        while error > limit:

            old_error = error
            delta = LearningRate(error, limit, parameters["delta"], ranges1, multiples1)
            coefs[epoch] = round(coefs[epoch] + delta, 2)

            sample_weights[len(train) - 1] = coefs[epoch]
            regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(regr,
                                                                                                                X_train,
                                                                                                                y_train,
                                                                                                                sample_weights)
            error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
            if parameters['wMAE']:
                mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
            else:
                mae = (np.sum(train_error) + error) / len(train)

            if not (parameters['nonC'] or parameters["nonOrder"]):
                if error == old_error and j == 1:
                    sys.exit(1)
                    
                if parameters['tolerance2']:
                    tolerance2 = parameters['tolerance2']
                    limit = round(epsilon + tolerance2, 6)

                j = 0
                while mae > limit:

                    old_mae = mae

                    regr = SVR(epsilon=epsilon, kernel="rbf", C=C, gamma=gamma)
                    regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(
                        regr, X_train, y_train, sample_weights)
                    error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
                    if parameters['wMAE']:
                        mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
                    else:
                        mae = (np.sum(train_error) + error) / len(train)

                    if mae > limit:
                        if mae == old_mae and j > 0:
                            sys.exit(1)
                        eta = LearningRate(error, limit, parameters["eta"], ranges1, multiples1)
                        C += eta

                    j += 1
            # else:
            #     if error == old_error:
            #         sys.exit(1)

        if error <= limit:
            if parameters['wMAE']:
                mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
            else:
                mae = (np.sum(train_error) + error) / len(train)
            sample_weights[len(train) - 1] = coefs[epoch]
            # print("Finish %s - %s" % (n, epoch))
            pbar.update(1)

    if parameters["nonOrder"]:
        if error == old_error and j == 1:
            sys.exit(1)

        if parameters['tolerance2']:
            tolerance2 = parameters['tolerance2']
            limit = round(epsilon + tolerance2, 6)
            
        j = 0
        while mae > limit:

            old_mae = mae

            regr = SVR(epsilon=epsilon, kernel="rbf", C=C, gamma=gamma)
            regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(
                regr, X_train, y_train, sample_weights)
            error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
            if parameters['wMAE']:
                mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
            else:
                mae = (np.sum(train_error) + error) / len(train)

            if mae > limit:
                if mae == old_mae and j > 0:
                    sys.exit(1)
                eta = LearningRate(error, limit, parameters["eta"], ranges1, multiples1)
                C += eta

            j += 1

    X_train = X[train, :]
    y_train = y[train]
    y_train_pred = regr.predict(X_train)
    X_validation = X[validation, :]
    y_validation = y[validation]
    y_validation_pred = regr.predict(X_validation)

    y_test_pred = regr.predict(X_test)

    result = []
    result.append(RealResults(data, majority_train_set))
    result.append(y_train[:majority_train_size])
    result.append(y_train_pred[:majority_train_size])

    result.append(RealResults(data, majority_validation_set))
    result.append(y_validation[:majority_validation_size])
    result.append(y_validation_pred[:majority_validation_size])

    result.append(RealResults(data, majority_test_set))
    result.append(y_test[:majority_test_size])
    result.append(y_test_pred[:majority_test_size])

    result.append(RealResults(data, train[majority_train_size:]))
    result.append(y_train[majority_train_size:])
    result.append(y_train_pred[majority_train_size:])

    result.append(RealResults(data, minority_validation_set))
    result.append(y_validation[majority_validation_size:])
    result.append(y_validation_pred[majority_validation_size:])

    result.append(RealResults(data, minority_test_set))
    result.append(y_test[majority_test_size:])
    result.append(y_test_pred[majority_test_size:])

    keys = RealResults(data, train[majority_train_size:])
    values = coefs
    result.append(keys)
    result.append(values)

    parameters["C"] = C
    result.append(list(parameters.keys()))
    result.append(list(parameters.values()))
    
    return result


def SIMLT(data: pd.DataFrame, X: np.array, y: np.array, split_data:list, n: int, parameters: list):
    epsilon, gamma, C, delta, eta, tolerance = (
        parameters["epsilon"],
        parameters["gamma"],
        parameters["C"],
        parameters["delta"],
        parameters["eta"],
        parameters["tolerance"]
    )
    
    majority_train_set = split_data[n - 1][0]
    majority_validation_set = split_data[n - 1][1]
    majority_test_set = split_data[n - 1][2]
    minority_train_set = split_data[n - 1][3]
    minority_validation_set = split_data[n - 1][4]
    minority_test_set = split_data[n - 1][5]

    majority_train_size = majority_train_set.shape[0]
    minority_train_size = minority_train_set.shape[0]
    majority_validation_size = majority_validation_set.shape[0]
    minority_validation_size = minority_validation_set.shape[0]
    majority_test_size = majority_test_set.shape[0]
    minority_test_size = minority_test_set.shape[0]
    # debug = self.debug

    train = majority_train_set
    validation = np.append(majority_validation_set, minority_validation_set)
    test = np.append(majority_test_set, minority_test_set)

    minority_train_num = len(minority_train_set)
    sample_weights = np.ones(len(train), dtype=float)
    coefs = np.ones(minority_train_num, dtype=float)
    minority_train_bak = minority_train_set

    regr = SVR(kernel="rbf", epsilon=epsilon, C=C, gamma=gamma)
    
    pbar = tqdm(total=minority_train_num)
    pbar.set_description("SIMLT (iteration %s)" % n)
    
    for epoch in range(minority_train_num):
        best = minority_train_set[epoch]
        X_train = X[train, :]
        y_train = y[train]
        X_best = X[best, :]
        y_best = y[best]
        sample_weights2 = np.append(sample_weights, coefs[epoch])
        
        regr = regr.fit(X_train, y_train, sample_weights)
        y_pred = regr.predict(X_best.reshape(1, -1))
        error = abs(y_pred - y_best)
    
        train2 = np.append(train, best)
        X_train = X[train2, :]
        y_train = y[train2]
        
        limit = round(epsilon + tolerance, 6)
        ranges1 = [i * limit for i in list(range(5,0,-1))]
        multiples1 = list(range(5,0,-1))
        
        while error > limit:

            old_error = error
            
            delta = LearningRate(error, limit, parameters["delta"], ranges1, multiples1)
            coefs[epoch] = round(coefs[epoch] + delta, 2)
            sample_weights2[len(train)] = coefs[epoch]
            
            regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(regr,
                                                                                                                X_train,
                                                                                                                y_train,
                                                                                                                sample_weights2)
            error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
        pbar.update(1)
        
    result = []
    keys = RealResults(data, minority_train_set)
    values = coefs
    result.append(minority_train_set)
    result.append(y[minority_train_set])
    result.append(keys)
    result.append(values)

    parameters["C"] = C
    result.append(list(parameters.keys()))
    result.append(list(parameters.values()))
    
    return result
              

def SIML(data: pd.DataFrame, X: np.array, y: np.array, split_data: list, parameters: list):

    iterations = split_data.__len__()
    results = []

    match parameters["wb"]:
        case True:
            if parameters["parallel"]:
                        results = (
                            Parallel(n_jobs=parameters["n_jobs"])
                            (delayed(SIMLT)(data, X, y, split_data, n, parameters)
                            for n in tqdm(range(iterations)))
                        )
            else:
                for n in range(1, len(split_data)+1, 1):
                    result = SIMLT(data, X, y, split_data, n, parameters)
                    results.append(result)
        case _:
            if parameters["parallel"]:
                        results = (
                            Parallel(n_jobs=parameters["n_jobs"])
                            (delayed(SIMLP)(data, X, y, split_data, n, parameters)
                            for n in tqdm(range(iterations)))
                        )
            else:
                for n in range(1, len(split_data)+1, 1):
                    result = SIMLP(data, X, y, split_data, n, parameters)
                    results.append(result)

    return results
    
    