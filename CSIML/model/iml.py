"""Imbalance learning module"""

import copy
import time
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from CSIML.analysis._base import index_to_name
from CSIML.analysis.common import evaluation
from CSIML.model.basis2 import Basis2Model


def train_model(
    regr, X_train: np.array, y_train: np.array, sample_weights: list
) -> Tuple:
    """Training ML models

    Args:
        regr (model): the basic ML model with hyperparameters.
        X_train (np.array): features training set.
        y_train (np.array): property traing set.
        sample_weights (list): weights for training instances.

    Returns:
        Tuple[model, list]: return trained ML model and metrics for training set.
    """
    regr = regr.fit(X_train, y_train, sample_weights)
    y_train_pred = regr.predict(X_train)
    metrics = evaluation(y_train, y_train_pred)
    return regr, metrics


def test_error(
    regr, X_test: np.array, y_test: np.array, test: np.array, method: str = "O1"
) -> Tuple[int, float, float]:
    """Test errors for rest minority training instances with trained ML model

    Following the ascend (smallest -> largest) or descend (largest -> smallest) for
    error or bandgap

    Args:
        regr (model): the basic ML model with hyperparameters.
        X_test (np.array): features of rest minority training set.
        y_test (np.array): property of rest minority training set.
        test (np.array): indexes of rest minority training set.
        method (str, optional): op method (one of "O1", "O2", "O3" and "O4").
            Defaults to "O1".

    Returns:
        Tuple[int, float, float]: return the index in tested instances, the index in
        all instances and the prediction value. (All indexes start from 0)
    """
    y_pred = regr.predict(X_test)
    errors = y_pred - y_test

    match method:
        case "O1":  # error_ascend
            index = np.argmin(abs(errors))
        case "O2":  # error_descend
            index = np.argmax(abs(errors))
        case "O3":  # bandgap_ascend
            index = np.argmin(y_test)
        case "O4":  # bandgap_descend
            index = np.argmax(y_test)

    return index, test[index], y_pred[index]


def learning_rate(
    error: float, min_error: float, step: float, ranges: list, multiples: list
) -> float:
    """Update the learning rate for the weights or costs according to rules

    Args:
        error (float): current prediction error comparing to experimental value.
        min_error (float): expected acceptable error.
        step (float): basic increment for one step (the hyperparameter ``delta``).
        ranges (list): error levels, the time of acceptable error.
        multiples (list): weight levels, the time of ``delta``.

    Returns:
        increment (float): return the increment for the weight to next step.
    """
    m = abs(error - min_error)
    i = int(len(ranges) / 2)
    t = -1
    while t == -1:
        if m > ranges[i - 1]:
            if i == 1:
                t = i
            elif m <= ranges[i - 2]:
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
    if error >= 0:
        increment = multiples[t - 1] * step
    else:
        increment = -multiples[t - 1] * step
    return increment


class IML(Basis2Model):
    """Imbalanced machine learning

    Args:
        data (pd.DataFrame): Initial dataset.
        basic_model (str, optional): the basic ML model. Defaults to "SVR".
        cv_method (str, optional): cross validation method. Defaults to "siml".
        parameters (dict, optional): parameters for basic ML model. It could include
            "epsilon", "gamma", "C" or other parameters. `epsilon` is the acceptable
            error for training, which is a small value usually.
        delta (float, optional): the increment step for weights. Defaults to 0.5.
        tolerance (float, optional): the tolerance for the error, helpful for numerical
            convergence. Defaults to 0.005.
        op_method (str, optional): optimization method for the order. Defaults to "O1".
            It supports "O1"(error ascending), "O2"(error descending),  "O3"(property
            ascending) and "O4"(property descending).
        ranges (np.array, optional): the times range matrix for the error. Defaults to
            np.array([5,4,3,2,1]).
        multiples (np.array, optional): the times range matrix for the weight increment
            corresponding to `ranges` . Defaults to np.array([5,4,3,2,1]).
        wMAE (bool, optional): whether using weighted MAE to evaluate. Defaults to False.
        DEBUG (bool, optional): whether using DEBUG mode, printing out the weight
            learning process. Defaults to False.
        CHECK (bool, optional): whether using CHECK mode for comparing weights. Defaults
            to False.
        mpi_mode (bool, optional): whether using MPI mode. Defaults to False.
        opt_C (bool, optional): whether optimizing hyperparameter C. Defaults to False.
        eta (float, optional): the increment step for hyperparameter C. Defaults to 1.0.
        C_max (float, optional): the maximum for hyperparameter C. Defaults to 50.
        C_limit (float, optional): the real acceptable error when hyperparameter C is
            optimized. Defaults to `self.limit`.

    Attributes:
        limit (float): the real acceptable error used in the calculation, which is the
            sum of `parameters["epsilon"]` and `tolerance`.
        increments (np.array): the weights matrix, which is times by `multiples` and
            `delta`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        basic_model: str = "SVR",
        cv_method: str = "siml",
        **kws,
    ) -> None:
        super().__init__(data, basic_model, cv_method, **kws)
        self.eta = kws.get("eta", 1.0)
        self.delta = kws.get("delta", 0.5)
        self.tolerance = kws.get("tolerance", 0.005)
        if "epsilon" in self.parameters:
            self.limit = round(self.parameters["epsilon"] + self.tolerance, 6)
        else:
            self.limit = round(0.1 + self.tolerance, 6)
        self.C_limit = kws.get("C_limit", self.limit)
        self.op_method = kws.get("op_method", "O1")
        self.ranges = kws.get("ranges", np.array([5, 4, 3, 2, 1]))
        self.multiples = kws.get("multiples", np.array([5, 4, 3, 2, 1]))
        self.increments = np.array([round(i * self.delta, 3) for i in self.multiples])
        self.wMAE = kws.get("wMAE", False)
        self.DEBUG = kws.get("DEBUG", False)
        self.CHECK = kws.get("CHECK", False)
        self.mpi_mode = kws.get("mpi_mode", False)
        self.opt_C = kws.get("opt_C", False)
        self.C_max = kws.get("C_max", 50)

    def iml_learn_train(self, regr, train: list, sample_weights: list) -> Tuple:
        """Imbalance Learning basic training function

        To train machine learning model with sample weights.

        Args:
            regr (model): the basic ML model with hyperparameters.
            train (list): instance indexes which will be used to train the ML model.
            sample_weights (list): weights of training instances.

        Returns:
            Tuple[model, float]: return trained model and prediciton error of last
            (newly added) training instance.
        """
        X = self.X
        y = self.y
        X_train = X[train, :]
        y_train = y[train]
        regr, metrics = train_model(regr, X_train, y_train, sample_weights)
        error = metrics["error"][-1]
        return regr, error

    def iml_learn_cliff(
        self,
        train: list,
        error: float,
        sample_weights: list,
        delta: float,
        old_coef: float,
        step: int,
    ) -> Tuple[float, float, int, float]:
        """Imbalance Learning special weight updating function for cliff

        Sometimes the weight updating has no change in a local area, which seems no hope
        to be converged. This situation is called as **cliff**. However, it's possible
        to be converged if we can give it a bigger weight (jump). This function is mainly
        for handling this problem by trying bigger weight.

        Args:
            train (list): training instance indexes including newly added minority
                instances.
            error (float): prediction error for last minority training instance.
            sample_weights (list): weights of training instances.
            delta (float): the step size for updating the weight.
            old_coef (float): old weight for last minority training instance.
            step (int): weight updating steps.

        Returns:
            Tuple[float, float, int, float]: return updated weight, prediction error,
            steps and new bigger increment for last minority training instance.
        """
        limit = self.limit
        DEBUG = self.DEBUG
        regr = self.regr
        increments = self.increments
        coef = 0.0

        # Optimize for the cliff (flat area)
        if DEBUG:
            print("Opt cliff -> ")

        increments = increments[increments > delta]
        k = 0
        while error > limit:
            if k > 0:
                delta = increments[0] * (k + 1)
                coef = round(old_coef + delta, 2)
                sample_weights[len(train) - 1] = coef
                regr, error = self.iml_learn_train(regr, train, sample_weights)
                step += 1
                if DEBUG:
                    print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")
                if error <= limit:
                    break
            else:
                for j in range(increments.size - 1, -1, -1):
                    delta = increments[0] * k + increments[j]
                    coef = round(old_coef + delta, 2)
                    sample_weights[len(train) - 1] = coef
                    regr, error = self.iml_learn_train(regr, train, sample_weights)
                    step += 1
                    if DEBUG:
                        print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")
                    if error <= limit:
                        break
            k += 1

        return coef, error, step, delta

    def iml_learn_track(
        self,
        train: list,
        error: float,
        sample_weights: list,
        delta: float,
        old_coef: float,
        step: int,
    ) -> Tuple[float, float, int]:
        """Imbalance Learning fast weight updating function

        Args:
            train (list): training instance indexes including newly added minority
                instances.
            error (float): prediction error for last minority training instance.
            sample_weights (list): weights of training instances.
            delta (float): the step size for updating the weight.
            old_coef (float): old weight for last minority training instance.
            step (int): weight updating steps.

        Returns:
            Tuple[float, float, int]: return updated weight, prediction error and steps
            for last minority training instance.
        """
        limit = self.limit
        regr = self.regr
        target_error = error
        DEBUG = self.DEBUG
        increments = self.increments
        coef_bak = old_coef + delta

        if DEBUG:
            print("Trackback -> ")
        if delta % increments[0] == 0:
            old_delta = delta - increments[0]
            increments = increments[increments <= increments[0]]
        else:
            delta_res = round(delta % increments[0], 2)
            old_delta = delta - delta_res
            increments = increments[increments <= delta_res]

        while increments.size > 2:
            middle_index = int(increments.size / 2)
            coef = old_coef + old_delta + increments[middle_index]
            sample_weights[len(train) - 1] = coef
            regr, error = self.iml_learn_train(regr, train, sample_weights)
            step += 1
            if error <= limit:
                increments = increments[middle_index:]
                target_error = error
            else:
                increments = increments[:middle_index]
            if DEBUG:
                print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")

        if increments.size == 2:
            coef = old_coef + old_delta + increments[1]
            sample_weights[len(train) - 1] = coef
            regr, error = self.iml_learn_train(regr, train, sample_weights)
            step += 1
            if error <= limit:
                coef = old_coef + old_delta + increments[1]
                sample_weights[len(train) - 1] = coef
                target_error = error
            else:
                coef = old_coef + old_delta + increments[0]
                sample_weights[len(train) - 1] = coef
                error = target_error
        elif increments.size == 1:
            coef = old_coef + old_delta + increments[0]
            sample_weights[len(train) - 1] = coef
            error = target_error
        else:
            # coef = old_coef + old_delta + increments[0]
            coef = coef_bak
            sample_weights[len(train) - 1] = coef
            error = target_error

        if DEBUG and coef != (coef_bak):
            print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")

        return coef, error, step

    def iml_learn(
        self,
        train: list,
        best: int,
        sample_weights: list,
        error: float,
    ) -> float:
        """Imbalance Learning iterative learning process function

        It's only for one minority training instance with existed training instances.

        Args:
            train (list): existed training instances indexes.
            best (int): global index of the best one in minority instances to be trained.
            sample_weights (list): weights of existed training instances.
            error (float): prediction error of the best one in minority instances to be
                trained.

        Returns:
            list: return the weight for the best instance.

        """
        coef = 1.0
        limit = self.limit

        old_coef = coef
        sample_weights = np.append(sample_weights, coef)
        train = np.append(train, best)
        X_train = self.X[train, :]
        y_train = self.y[train]
        regr, _ = train_model(self.regr, X_train, y_train, sample_weights)

        ranges = [round(i * limit, 3) for i in self.ranges]
        multiples = self.multiples

        step = 0
        old_delta = 0
        IS_FLAT = False

        if self.DEBUG:
            start_at = time.time()

        while error > limit:
            old_coef = coef
            delta = learning_rate(error, limit, self.delta, ranges, multiples)
            old_delta = delta
            coef = round(old_coef + delta, 2)
            sample_weights[len(train) - 1] = coef
            step += 1
            regr, error = self.iml_learn_train(regr, train, sample_weights)

            if self.DEBUG:
                print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")

            if old_delta == delta and delta <= 1.0 and error > limit:
                IS_FLAT = True
                coef, error, step, delta = self.iml_learn_cliff(
                    train, error, sample_weights, delta, old_coef, step
                )

        # Trackback
        if self.DEBUG and IS_FLAT and delta <= self.increments[0]:
            print("")
        else:
            coef, error, step = self.iml_learn_track(
                train, error, sample_weights, self.delta, old_coef, step
            )

        if error <= limit:
            sample_weights[len(train) - 1] = coef
            if self.DEBUG:
                end_at = time.time()
                print(
                    f"END (coef: {coef:.2f}, "
                    f"error: {(error - limit):.6f}, step: {step}, "
                    f"used time: {(end_at - start_at):.3f}s)"
                )

        return coef

    def __fit__(self, df: list) -> np.array:
        """Imbalance Learning main function

        Args:
            df (np.array): split dataset of instance indexes, including majority
                training, majority validation, majority test, minority training, minority
                validation and minority test sets (totally 6 sets).

        Returns:
            result (np.array): return prediction result for one dataset.
        """
        majority_train_set = df[0]
        minority_train_set = df[3]
        minority_train_size = df[3].shape[0]
        train = majority_train_set
        sample_weights = np.ones(len(train), dtype=float)
        coefs = np.ones(minority_train_size, dtype=float)
        order = []

        if self.show_tips:
            pbar = tqdm(total=minority_train_size)
            pbar.set_description("IML (Fit)")

        for epoch in range(minority_train_size):
            X_train = self.X[train, :]
            y_train = self.y[train]
            regr, _ = train_model(self.regr, X_train, y_train, sample_weights)
            if self.CHECK:
                best = minority_train_set[epoch]
                best_pred = regr.predict(self.X)[best]
            else:
                best_index, best, best_pred = test_error(
                    regr,
                    self.X[minority_train_set, :],
                    self.y[minority_train_set],
                    minority_train_set,
                    self.op_method,
                )
            y_best = self.y[best]
            order = np.append(order, int(best))
            error = abs(y_best - best_pred)

            if self.DEBUG:
                print(f"{best:3}: 1.00({(error-self.limit):.6f}) -> ")

            coef = self.iml_learn(train, best, sample_weights, error)
            coefs[epoch] = coef

            if not self.CHECK:
                sample_weights = np.append(sample_weights, coef)
                train = np.append(train, best)
            minority_train_set = np.delete(minority_train_set, best_index)
            if self.show_tips:
                pbar.update(1)

        X_train = self.X[train, :]
        y_train = self.y[train]
        regr, metrics = train_model(self.regr, X_train, y_train, sample_weights)
        if self.opt_C:
            C = self.iml_learn_hyper(X_train, y_train, metrics["error"], coefs)
        else:
            C = self.parameters["C"]
        regr = [C, order.astype(int), coefs]
        return regr

    def iml_learn_hyper(
        self, X_train: list, y_train: list, errors: list, coefs: list
    ) -> float:
        """Update hyperparameter C for imbalance learning

        Args:
            X_train (list): the features (input) of the training set.
            y_train (list): the property (output) of the training set.
            errors (list): the errors for all instances in the training set.
            coefs (list): the coefficients(weights) for all instances in the training set.

        Returns:
            float: return updated hyperparameter C.
        """
        C = copy.deepcopy(self.parameters["C"])
        regr = copy.deepcopy(self.regr)
        mae = self.get_avg_error(errors, coefs)
        sample_weights = np.ones(len(errors) - len(coefs), dtype=float)
        sample_weights = np.append(sample_weights, coefs)
        j = 0
        while mae > self.C_limit:
            old_mae = mae
            C += self.eta
            if C > self.C_max:
                break
            regr.C = C
            regr, metrics = train_model(regr, X_train, y_train, sample_weights)
            mae = self.get_avg_error(metrics["error"], coefs)
            if mae == old_mae and j > 0:
                break
            j += 1
        return C

    def get_avg_error(self, errors: np.array, coefs: list) -> float:
        """Get (weighted) mean average error

        Args:
            errors (np.array): the errors for all training instances.
            coefs (list): the coefficients(weights) for all training instances.

        Returns:
            float: return (weighted) mean average error (MAE or wMAE).
        """
        if self.wMAE:
            return (sum(errors[: -len(coefs)]) + coefs * errors[-len(coefs) :]) / len(
                errors
            )
        return errors.mean()

    def __predict__(self, df: pd.DataFrame, regr) -> list:
        """Predict all instances in one iteration with trained ML model

        Args:
            df (pd.DataFrame): the dataset in one iteration of cross validaton.
            regr: single trained ML model.

        Returns:
            list: return prediction result.
        """
        train = np.append(df[0], df[3])
        X_train = self.X[train, :]
        y_train = self.y[train]
        regr_c = copy.deepcopy(regr)
        sample_weights = np.ones(len(df[0]), dtype=float)
        sample_weights = np.append(sample_weights, regr_c[2])
        regr = self.regr
        regr.C = regr_c[0]
        regr, _ = train_model(regr, X_train, y_train, sample_weights)
        result = []
        if not self.CHECK:
            y_pred = regr.predict(self.X)
            for dfi in df:
                result = (
                    result
                    + [index_to_name(self.data, dfi)]
                    + [self.y[dfi]]
                    + [y_pred[dfi]]
                )
        result.append(regr_c[1])
        result.append(self.y[regr_c[1]])
        keys = index_to_name(self.data, regr_c[1])
        values = regr_c[2]
        result.append(keys)
        result.append(values)
        parameters = self.parameters
        parameters["C"] = regr_c[0]
        result.append(list(parameters.keys()))
        result.append(list(parameters.values()))
        return result
