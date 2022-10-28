import math
from random import seed, sample

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from tabulate import tabulate
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def KFoldSize(majority_size: dict, minority_size: dict, fold: int = 10):
    """

    :param majority_size:
    :param minority_size:
    :param fold: 10 or 5 (for pls and jpcl CV)
    :return: majority_size, minority_size
    """
    if majority_size["Total"] % fold == 0:
        majority_size["Validation"] = math.ceil(majority_size["Total"] * 0.1)
        majority_size["Train"] = majority_size["Total"] - majority_size["Validation"]
    else:
        majority_size["Validation"] = math.ceil(majority_size["Total"] * 0.1)
        majority_size["Train"] = majority_size["Total"] - majority_size["Validation"]
        majority_size["Validation"] = (
            str(majority_size["Validation"])
            + " or "
            + str(majority_size["Validation"] - 1)
        )
        majority_size["Train"] = (
            str(majority_size["Train"]) + " or " + str(majority_size["Train"] + 1)
        )
    if minority_size["Total"] % fold == 0:
        minority_size["Validation"] = math.ceil(minority_size["Total"] * 0.1)
        minority_size["Train"] = minority_size["Total"] - minority_size["Validation"]
    else:
        minority_size["Validation"] = math.ceil(minority_size["Total"] * 0.1)
        minority_size["Train"] = minority_size["Total"] - minority_size["Validation"]
        minority_size["Validation"] = (
            str(minority_size["Validation"])
            + " or "
            + str(minority_size["Validation"] - 1)
        )
        minority_size["Train"] = (
            str(minority_size["Train"]) + " or " + str(minority_size["Train"] + 1)
        )
    return majority_size, minority_size

def PrintSize(majority_size: dict, minority_size: dict):
    d = [["Majority", 0, 0, 0, 0], ["Minority", 0, 0, 0, 0]]
    d[0][1] = majority_size["Train"]
    d[0][2] = majority_size["Validation"]
    d[0][3] = majority_size["Test"]
    d[0][4] = majority_size["Total"]
    d[1][1] = minority_size["Train"]
    d[1][2] = minority_size["Validation"]
    d[1][3] = minority_size["Test"]
    d[1][4] = minority_size["Total"]

    print("")
    print(tabulate(d, headers=["Size", "Train", "Validation", "Test", "Total"]))

def CalSize(
    data: pd.DataFrame,
    cv_method: str,
    sampling_method: str = "None",
    majority_pro: float = 0.2,
    minority_pro: float = 0.2,
    threshold: float = 5.0,
    majority_test_num: int = 24,
):
    """

    :param data: source data
    :param cv_method: siml, pls, jpcl, basis1
            siml: (4:1)4:1 (5-fold CV, 5x5x5x5=625 iterations)
            pls:  1,2,3,4,5,6,7,... (10-fold CV, 10 iterations)
            jpcl: 1,2,3,4,5,1,2,3,4,5,... (5-fold CV, 5 iterations)
            basis1: 24 (test), 4:1(Train vs Validation) (5-fold CV, 5 iterations)
    :param majority_pro:
    :param minority_pro:
    :param threshold:
    :param majority_test_num:
    :return:
    """
    data_size = data.shape[0]
    majority_size = dict(Total=0, Train=0, Validation=0, Test=0)
    minority_size = dict(Total=0, Train=0, Validation=0, Test=0)

    majority_size["Total"] = len(data[data["Experimental"] < threshold])
    minority_size["Total"] = data_size - majority_size["Total"]

    match cv_method:
        case "siml":
            majority_size["Test"] = math.ceil(majority_size["Total"] * majority_pro)
            majority_rest_size = majority_size["Total"] - majority_size["Test"]
            majority_size["Validation"] = math.ceil(majority_rest_size * majority_pro)
            majority_size["Train"] = majority_rest_size - majority_size["Validation"]

            minority_size["Test"] = math.ceil(minority_size["Total"] * minority_pro)
            minority_rest_size = minority_size["Total"] - minority_size["Test"]
            minority_size["Validation"] = math.ceil(minority_rest_size * minority_pro)
            minority_size["Train"] = minority_rest_size - minority_size["Validation"]

        case "pls":
            majority_size, minority_size = KFoldSize(majority_size, minority_size)

        case "jpcl":
            majority_size, minority_size = KFoldSize(majority_size, minority_size, 5)

        case "basis1":
            minority_size["Test"] = minority_size["Total"]
            majority_size["Test"] = majority_test_num
            majority_rest_size = majority_size["Total"] - majority_size["Test"]
            majority_size["Validation"] = math.ceil(majority_rest_size * majority_pro)
            majority_size["Train"] = majority_rest_size - majority_size["Validation"]

    if sampling_method == "oversampling" and cv_method == "siml":
        minority_size["Total"] = (
            minority_size["Total"] - minority_size["Train"] + majority_size["Train"]
        )
        minority_size["Train"] = (
            str(majority_size["Train"]) + "(" + str(minority_size["Train"]) + ")"
        )

    PrintSize(majority_size, minority_size)
    return majority_size, minority_size

def SplitData(
    data: pd.DataFrame,
    cv_method: str,
    sampling_method: str = "None",
    majority_pro: float = 0.2,
    minority_pro: float = 0.2,
    random_state: int = 3,
    threshold: float = 5.0,
    majority_test_num: int = 24,
):
    """

    :param data:
    :param cv_method:
    :param sampling_method:
    :param majority_pro:
    :param minority_pro:
    :param threshold:
    :param majority_test_num:
    :return: split_data
    """

    split_data = []
    data_size = data.shape[0]
    majority_size = len(data[data["Experimental"] < threshold])
    minority_size = data_size - majority_size
    majority_set = np.arange(majority_size)
    minority_set = np.arange(data_size)[-minority_size:]

    match cv_method:
        case "siml":
            split_data = SplitDataSIML(
                data,
                majority_set,
                minority_set,
                random_state=random_state,
                sampling_method=sampling_method,
            )

        case "pls":
            split_data = SplitDataKFoldCV(
                data,
                majority_set,
                minority_set,
                random_state=random_state,
                sampling_method=sampling_method,
            )

        case "jpcl":
            split_data = SplitDataKFoldCV(
                data,
                majority_set,
                minority_set,
                5,
                random_state=random_state,
                sampling_method=sampling_method,
            )

        case "basis1":
            split_data = SplitDataBasis(
                majority_set, minority_set, random_state=random_state
            )

        case "op":
            split_data = SplitDataOPKFoldCV(
                data,
                majority_set,
                minority_set,
                random_state=random_state,
                sampling_method=sampling_method,
            )

    print("Split data: OK")
    return split_data

def SplitDataSIML(
    data: pd.DataFrame,
    majority_set: np.array,
    minority_set: np.array,
    majority_pro: float = 0.2,
    minority_pro: float = 0.2,
    random_state: int = 3,
    sampling_method: str = "None",
):
    """

    :param majority_set: majority instances indexes
    :param minority_set: minority instances indexes
    :param majority_pro: majority proportion, default: 0.2
    :param minority_pro: minority proportion, default: 0.2
    :param random_state: the seed of random, default: 3
    :return: spilt_data: the spiltted data
    """
    ss1 = ShuffleSplit(
        n_splits=int(1 / majority_pro),
        test_size=majority_pro,
        random_state=random_state,
    )
    ss2 = ShuffleSplit(
        n_splits=int(1 / minority_pro),
        test_size=minority_pro,
        random_state=random_state,
    )
    split_data = []

    for minority_list, minority_test in ss2.split(minority_set):
        for minority_train, minority_validation in ss2.split(minority_list):
            for majority_list, majority_test in ss1.split(majority_set):
                for majority_train, majority_validation in ss1.split(majority_list):
                    cv_data = []

                    # Fix a bug for indexes
                    majority_train_set = majority_set[majority_list[majority_train]]
                    majority_validation_set = majority_set[
                        majority_list[majority_validation]
                    ]
                    majority_test_set = majority_set[majority_test]
                    minority_train_set = minority_set[minority_list[minority_train]]
                    minority_validation_set = minority_set[
                        minority_list[minority_validation]
                    ]
                    minority_test_set = minority_set[minority_test]

                    if sampling_method == "oversampling":
                        y = data.iloc[:, 1].values
                        train = np.append(majority_train_set, minority_train_set)
                        c = y[train] <= 5
                        train = train[:, np.newaxis]
                        train_resample, c = OverSampling(train, c)
                        train_resample = train_resample.flatten()
                        minority_train_set = train_resample[len(majority_train_set) :]
                    elif sampling_method == "undersampling":
                        y = data.iloc[:, 1].values
                        train = np.append(majority_train_set, minority_train_set)
                        c = y[train] <= 5
                        train = train[:, np.newaxis]
                        train_resample, c = UnderSampling(train, c)
                        train_resample = train_resample.flatten()
                        majority_train_set = train_resample[len(minority_train_set) :]

                    # li0 = 467 in minority_set
                    # li1 = 467 in minority_train_set
                    # li2 = 467 in minority_validation_set
                    # li3 = 467 in minority_test_set

                    cv_data.append(majority_train_set)
                    cv_data.append(majority_validation_set)
                    cv_data.append(majority_test_set)
                    cv_data.append(minority_train_set)
                    cv_data.append(minority_validation_set)
                    cv_data.append(minority_test_set)

                    split_data.append(cv_data)
    return split_data

def SplitDataBasis(
    majority_set: np.array,
    minority_set: np.array,
    majority_test_num: int = 24,
    majority_pro: float = 0.2,
    random_state: int = 3,
):
    """

    :param majority_set: majority instances indexes
    :param minority_set: minority instances indexes
    :param majority_pro: majority proportion, default: 0.2
    :param majority_test_num: the test num for majority, default: 24
    :param random_state: the seed of random, default: 3
    :return: spilt_data: the spiltted data
    """
    ss1 = ShuffleSplit(
        n_splits=int(1 / majority_pro),
        test_size=majority_pro,
        random_state=random_state,
    )
    split_data = []

    majority_test = sample(list(majority_set), majority_test_num)
    majority_set = list(set(majority_set) - set(majority_test))
    minority_test = minority_set
    test_set = np.append(majority_test, minority_test)

    for majority_train, majority_validation in ss1.split(majority_set):
        cv_data = []

        cv_data.append(majority_train)
        cv_data.append(majority_validation)
        cv_data.append(np.array(majority_test))
        cv_data.append(minority_test)

        split_data.append(cv_data)

    return split_data


def SplitDataKFoldCV(
    data: pd.DataFrame,
    majority_set: np.array,
    minority_set: np.array,
    fold: int = 10,
    random_state: int = 3,
    sampling_method: str = "None",
):
    """

    :param majority_set: majority instances indexes
    :param minority_set: minority instances indexes
    :param fold: the fold for CV, default: 10
    :return: split_data: the spiltted data
    """
    split_data = []

    for i in range(fold):
        cv_data = []

        majority_validation_set = majority_set[majority_set % fold == i]
        majority_test_set = majority_validation_set
        majority_train_set = np.delete(majority_set, majority_validation_set)
        minority_indexes = np.arange(minority_set.shape[0])
        minority_validation_indexes = minority_indexes % fold == i
        minority_validation_set = minority_set[minority_validation_indexes]
        minority_test_set = minority_validation_set
        minority_train_set = np.delete(minority_set, minority_validation_indexes)

        if sampling_method == "oversampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train_set, minority_train_set)
            c = y[train] <= 5
            train = train[:, np.newaxis]
            train_resample, c = OverSampling(train, c, random_state=random_state)
            train_resample = train_resample.flatten()
            minority_train_set = train_resample[len(majority_train_set) :]
        elif sampling_method == "undersampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train_set, minority_train_set)
            c = y[train] <= 5
            train = train[:, np.newaxis]
            train_resample, c = UnderSampling(train, c, random_state=random_state)
            train_resample = train_resample.flatten()
            majority_train_set = train_resample[len(minority_train_set) :]

        cv_data.append(majority_train_set)
        cv_data.append(majority_validation_set)
        cv_data.append(majority_test_set)
        cv_data.append(minority_train_set)
        cv_data.append(minority_validation_set)
        cv_data.append(minority_test_set)

        split_data.append(cv_data)

    return split_data


def SplitDataOPKFoldCV(
    data: pd.DataFrame,
    majority_set: np.array,
    minority_set: np.array,
    fold: int = 10,
    random_state: int = 3,
    sampling_method: str = "None",
):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        majority_set (np.array): _description_
        minority_set (np.array): _description_
        fold (int, optional): _description_. Defaults to 10.
        sampling_method (str, optional): _description_. Defaults to "None".

    Returns:
        _type_: _description_
    """
    split_data = []

    # Obtain test_set for majority and minority
    seed(a=random_state)
    # seed(a=10)
    # seed(a=30)
    majority_pro = round(1 / fold, 1)
    minority_pro = round(1 / fold, 1)
    majority_size, minority_size = CalSize(
        data, "siml", majority_pro=majority_pro, minority_pro=minority_pro
    )
    majority_test_set = sample(list(majority_set), majority_size["Test"])
    majority_set = np.array(list(set(majority_set) - set(majority_test_set)))
    minority_test_set = sample(list(minority_set), minority_size["Test"])
    minority_set = np.array(list(set(minority_set) - set(minority_test_set)))
    majority_test_set = np.array(majority_test_set)
    minority_test_set = np.array(minority_test_set)

    # Obtain training_set and validation_set for majority and minority
    for i in range(fold):
        cv_data = []

        majority_indexes = np.arange(majority_set.shape[0])
        majority_validation_indexes = majority_indexes % fold == i
        majority_validation_set = majority_set[majority_validation_indexes]
        majority_train_set = np.delete(majority_set, majority_validation_indexes)
        minority_indexes = np.arange(minority_set.shape[0])
        minority_validation_indexes = minority_indexes % fold == i
        minority_validation_set = minority_set[minority_validation_indexes]
        minority_train_set = np.delete(minority_set, minority_validation_indexes)

        if sampling_method == "oversampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train_set, minority_train_set)
            c = y[train] <= 5
            train = train[:, np.newaxis]
            train_resample, c = OverSampling(train, c, random_state)
            train_resample = train_resample.flatten()
            minority_train_set = train_resample[len(majority_train_set) :]
        elif sampling_method == "undersampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train_set, minority_train_set)
            c = y[train] <= 5
            train = train[:, np.newaxis]
            train_resample, c = UnderSampling(train, c, random_state)
            train_resample = train_resample.flatten()
            majority_train_set = train_resample[len(minority_train_set) :]

        cv_data.append(majority_train_set)
        cv_data.append(majority_validation_set)
        cv_data.append(majority_test_set)
        cv_data.append(minority_train_set)
        cv_data.append(minority_validation_set)
        cv_data.append(minority_test_set)

        split_data.append(cv_data)

    return split_data


def OverSampling(X: pd.DataFrame, y: pd.DataFrame, random_state: int = 3):
    """

    :param X: features
    :param y: property
    :param random_state: the seed for random process
    :return:
    """
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled


def UnderSampling(X: pd.DataFrame, y: pd.DataFrame, random_state: int = 3):
    """

    :param X:
    :param y:
    :param random_state:
    :return:
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    return X_resampled, y_resampled
