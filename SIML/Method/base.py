from multiprocessing import Pool

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, \
    mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from SIML.Analytics.analysis import RealResults
import SIML.CV.cv as cv
import sys, pprint


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

    return results


def PredictWithMinor(data: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, split_data: pd.DataFrame, parameters: dict,
                     basic_model: str = "SVR", parallel: bool = False, processes: int = 4):
    if parallel:
        pool = Pool(processes)
        # for res in tqdm(
        #         pool.imap_unordered(PredictBandgap, range(1, iterations + 1)),
        #         total=iterations,
        #         bar_format="Basis2: {percentage:3.0f}% {bar}{r_bar}",
        # ):
        #     results.append(res[0][0])
        #     v_results.append(res[1][0])
        #     t_results.append(res[2][0])
        #     errors.append(res[3][0])
        #     o_results.append(res[4][0])
        pool.close()
        pool.join()
    else:
        results = []

        # pbar = tqdm(total=iterations)
        # pbar.set_description("Basis 2")

        for df in split_data:
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
            # result.append(RealResults(data, train_resample[len(df[0]):]))
            # result.append(y_train[len(df[0]):])
            result.append(y_train_pred[len(df[0]):])

            result.append(RealResults(data, df[4]))
            result.append(y[df[4]])
            result.append(y_validation_pred[len(df[1]):])

            result.append(RealResults(data, df[5]))
            result.append(y[df[5]])
            result.append(y_test_pred[len(df[2]):])

            result.append(Counter(df[3]))

            results.append(result)

    print("OK")

    return results


def TrainModel(regr, X_train, y_train, sample_weights):
    regr = regr.fit(X_train, y_train, sample_weights)
    y_train_pred = regr.predict(X_train)

    error, mae, variance, rmse, mape, r2, std = Evaluation(
        y_train, y_train_pred
    )

    return regr, error, mae, variance, rmse, mape, r2, std


def TestError(regr, X_test, y_test, test, method: str="error_ascend"):
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
        case "error_ascend":
            index = np.argmin(abs(test_error))
        case "error_descend":
            index = np.argmax(abs(test_error))
        case "bandgap_ascend":
            index = np.argmin(y_test)
        case "bandgap_descend":
            index = np.argmax(y_test)

    return index, test[index], y_pred[index]


def SIML(data: pd.DataFrame, X: np.array, y: np.array, split_data: list, parameters: list, op_method: str="error_ascend"):
    epsilon, gamma, C, delta, eta, tolerance = (
        parameters["epsilon"],
        parameters["gamma"],
        parameters["C"],
        parameters["delta"],
        parameters["eta"],
        parameters["tolerance"]
    )

    results = []

    for n in range(1, len(split_data)+1, 1):

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

            while error > (epsilon + tolerance):

                old_error = error
                coefs[epoch] = round(coefs[epoch] + delta, 2)

                sample_weights[len(train) - 1] = coefs[epoch]
                regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(regr,
                                                                                                                  X_train,
                                                                                                                  y_train,
                                                                                                                  sample_weights)
                error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
                mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)

                if parameters['nonC']:
                    if error == old_error and j == 1:
                        sys.exit(1)

                    j = 0
                    while mae > (epsilon + tolerance):

                        old_mae = mae

                        regr = SVR(epsilon=epsilon, kernel="rbf", C=C, gamma=gamma)
                        regr, train_error, train_mae, train_var, train_rmse, train_mape, train_r2, train_std = TrainModel(
                            regr, X_train, y_train, sample_weights)
                        error = abs(y_best - regr.predict(X_best.reshape(1, -1)))
                        mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)

                        if mae > (epsilon + tolerance):
                            if mae == old_mae and j > 0:
                                sys.exit(1)
                            C += eta

                        j += 1
                else:
                    if error == old_error:
                        sys.exit(1)

            if error <= (epsilon + tolerance):
                mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
                sample_weights[len(train) - 1] = coefs[epoch]
                print("Finish %s - %s" % (n, epoch))

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

        # result.append(coefs)
        keys = RealResults(data, train[majority_train_size:])
        values = coefs
        counters = dict(zip(keys, values))
        result.append(counters)

        parameters["C"] = C
        result.append(parameters)

        results.append(result)

    return results
