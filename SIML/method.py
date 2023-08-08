import pprint
import sys
import time
from collections import Counter, defaultdict
from functools import wraps
from multiprocessing import Pool
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, wrap_non_picklable_objects
from sklearnex import patch_sklearn

patch_sklearn()
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from SIML.analysis import evaluation, index_to_name
from SIML.cv import DatasetSize


# @delayed
# @wrap_non_picklable_objects
def timer(func):
    """A wrapper to calculate the cpu time required by function excution

    Args:
        func (function): the name of function.

    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f"{func.__name__} cost time {time_spend} s")
        return result

    return func_wrapper


class SIML(DatasetSize):
    """The class for training ML models, especially SIML model

    Note:
        All input parameters are saved as attributes of the class.

    Args:
        data (pd.DataFrame): the dataset with features and property.
        method (str, optional): the method name, supporting "basis1", "basis2" and 
            "siml". Defaults to "siml".
        basic_model (str, optional): the name of basic model, supporting "SVR", "DT" 
            and "RF". Defaults to "SVR".
        cv_method (str, optional): the cross validation method, supporting "basis1", 
            "pls", "jpcl" and "siml". Defaults to "siml" if None or not one of them.
        sampling_method (srt, optional): the resampling method, supporting "None", 
            "oversampling" and "undersampling". Defaults to None.
        threshold (float, optional): the value to split majority and minority. Defaults 
            to 5.0 (for bandgap).
        n_jobs (int, optional): the number of desired processors for parallel. It will 
            be ignored if MPI is used. Defaults to 1.
        eta (float, optional): the step size for updating hyperparameters. It's for ``C`` 
            if SVR. Defaults to 1.0.
        delta (float, optional): the step size for updating the weight. Defaults to 0.5.
        tolerance (float, optional): the tolerance for acceptable error, helpful for 
            converage. Defaults to 0.005.
        op_method (str, optional): the optimization order in SIML, including "O1"~"O4". 
            They are corresponding to "error_ascend", "error_descend", "property_ascend" 
            and "property_descend". Defaults to "O1".
        ranges (list, optional): error levels, the time of acceptable error. Defaults to 
            [5, 4, 3, 2, 1].
        multiples (list, optional): weight levels, the time of ``delta``. Defaults to 
            [5, 4, 3, 2, 1].
        random_state (int, optional): the random seed for sampling process. 
            Defaults to 3.
        wMAE (bool, optional): using weighted MAE if true, otherwise normal MAE. 
            Defaults to False.
        DEBUG (bool, optional): enable DEBUG mode if true, otherwise disable. Defaults 
            to False.
        CHECK (bool, optional): enable checking the process of learning weight if true, 
            otherwise disable. Defaults to False.

    Attributes:
        increments (np.array): real vaules of increments for the weight.
        parameters (dict): hyperparameters for basic models.
        limit (float): the sum of ``epsilon`` in parameters and ``tolerance``. The
            ``epsilon`` will be 0.1 if not found in parameters.

    Examples:
        >>> from SIML.method import SIML
        >>>
        >>> data = pd.read_excel("./examples/bandgap.xlsx", "468")
        >>> parameters = dict({
                "C": 10,
                "gamma": 0.01,
                "epsilon": 0.2,
            })
        >>> ranges = multiples = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]

        >>> # Basis1
            basis1 = SIML(data, "basis1", cv_method="basis1", parameters=parameters)

        >>> # Basis2
            basis2 = SIML(data, "basis2", parameters=parameters)

        >>> # SIML
            siml = SIML(
                data,
                "siml",
                cv_method="op",
                ranges=ranges,
                multiples=multiples,
                n_jobs=10,
                DEBUG=True,
                CHECK=True,
                parameters=parameters
            )

    """
    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "siml",
        basic_model: str = "SVR",
        cv_method: str = "siml",
        sampling_method: str = None,
        threshold: float = 5.0,
        n_jobs: int = 1,
        eta: float = 1.0,
        delta: float = 0.5,
        tolerance: float = 0.005,
        op_method: str = "O1",
        ranges: np.array = [5, 4, 3, 2, 1],
        multiples: np.array = [5, 4, 3, 2, 1],
        random_state: int = 3,
        wMAE: bool = False,
        DEBUG: bool = False,
        CHECK: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(data, cv_method, sampling_method, threshold)
        self.method = method
        self.basic_model = basic_model
        self.eta = eta
        self.delta = delta
        self.tolerance = tolerance
        self.op_method = op_method
        self.ranges = ranges
        self.multiples = multiples
        self.random_state = random_state
        self.wMAE = wMAE
        self.DEBUG = DEBUG
        self.CHECK = CHECK
        self.increments = np.array([round(i * self.delta, 3) for i in multiples])
        self.parameters = defaultdict(int)

        if kwargs is not None:
            try:
                for key, value in kwargs["parameters"].items():
                    self.parameters[key] = value
            except ValueError:
                return None

        if n_jobs == 0:
            print(f"'n_jobs' cannot be 0\nIt will be forced into 1")
            n_jobs = 1
        self.n_jobs = n_jobs
        if self.parameters["epsilon"]:
            self.limit = round(self.parameters["epsilon"] + tolerance, 6)
        else:
            self.limit = round(0.1 + tolerance, 6)
        parameters = dict(self.parameters)
        match basic_model := self.basic_model:
            case "DT":
                regr = DecisionTreeRegressor(max_depth=10)
            case "RF":
                regr = RandomForestRegressor(n_estimators=10)
            case _:  # Default model is SVR
                regr = SVR(kernel="rbf", C=10)

        if parameters is not None:
            for key, value in parameters.items():
                regr.__setattr__(key, value)

        self.regr = regr

    def fit_predict(self, **kws) -> list:
        """Fit ML models with data and settings, as well as predicting

        Returns:
            list: return prediction results for training, validation and test set of
            majority and minority respectively in different cross validation
            iterations. For ``SIML`` model, it also contain weights for minority
            training instances.

        Note:
            ``basis1`` method can be only used with "basis1" cv method. If not, the cv
            method will be forced to "basis1".

        """
        method = self.method
        if method == "basis1" and self.cv_method != "basis1":
            print(
                f"When using 'basis1' method, CV method must be 'basis1' too. "
                f"(Now it's '{self.cv_method}')"
                f"It will be forced to 'basis1' automatically."
            )
            self.cv_method = "basis1"

        self.cal_size()
        self.split_data(random_state=self.random_state, **kws)
        self.preprocessing(**kws)

        results = []

        match method:
            case "basis1":
                results = self.predict_only_major()
            case "basis2":
                results = self.predict_with_minor()
            case "siml":
                results = self.iml()

        return results

    def preprocessing(self, ascending=False) -> None:
        """Data preprocessing

        It mainly includes:
            * reordering the data according to the property value ascendly.
            * normalizing the features with MinMaxScaler().

        It's optional. For bandgaps with features based on elemetns, it's needed.

        """
        data = self.data
        if ascending:
            data.sort_values(by="Experimental", inplace=True, ascending=True)
        X = data.iloc[:, 3:].fillna(0).values
        X = MinMaxScaler().fit_transform(pd.DataFrame(X))
        y = data.iloc[:, 1].values
        self.X = X
        self.y = y

    @timer
    def predict_only_major(self) -> list:
        """Train ML models only with majority set and predict all instances

        It is also called as ``basis1`` method.

        Returns:
            results (list): return prediction results.

        """
        data = self.data
        splited_data = self.splited_data
        X = self.X
        y = self.y
        regr = self.regr
        iterations = self.iterations
        results = []

        pbar = tqdm(total=iterations)
        pbar.set_description("Basis 1")
        for dataset in splited_data:
            train = dataset[0]
            validation = dataset[1]
            majority_test = dataset[2]
            minority_test = dataset[3]
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
            train = index_to_name(data, train)
            validation = index_to_name(data, validation)
            majority_test = index_to_name(data, majority_test)
            minority_test = index_to_name(data, minority_test)

            result = (
                [train]
                + [y_train]
                + [y_train_pred]
                + [validation]
                + [y_validation]
                + [y_validation_pred]
                + [majority_test]
                + [y_majority_test]
                + [y_majority_test_pred]
                + [minority_test]
                + [y_minority_test]
                + [y_minority_test_pred]
            )

            results.append(result)
            pbar.update(1)

        return results

    def __predict__(self, df: pd.DataFrame) -> np.array:
        """Training and prediction function for ``basis2`` method

        It trains the ML model with majority and minority instances.

        Args:
            df (pd.DataFrame): dataset with features and property.

        Returns:
            np.array: prediction result.

        """
        data = self.data
        X = self.X
        y = self.y
        regr = self.regr
        result = []

        train = np.append(df[0], df[3])
        X_train = X[train, :]
        y_train = y[train]

        regr = regr.fit(X_train, y_train)
        y_pred = regr.predict(X)

        for dfi in df:
            result = result + [index_to_name(data, dfi)] + [y[dfi]] + [y_pred[dfi]]

        counts = Counter(df[3])
        result.append(index_to_name(data, list(counts.keys())))
        result.append(counts.values())

        # result.append(list(parameters.keys()))
        # result.append(list(parameters.values()))
        return result

    @timer
    def predict_with_minor(self) -> np.array:
        """Train ML models with all sets and predict all instances

        It is also called as ``basis2`` method.

        Returns:
            results (list): return prediction results.

        """
        splited_data = self.splited_data
        iterations = self.iterations
        n_jobs = self.n_jobs

        if n_jobs == 1:
            results = []
            pbar = tqdm(total=iterations)
            pbar.set_description("Basis 2")
            for dataset in splited_data:
                result = self.__predict__(dataset)
                results.append(result)
                pbar.update(1)
        else:
            results = Parallel(n_jobs)(
                delayed(self.__predict__)(splited_data[i])
                for i in tqdm(range(iterations))
            )
            # delayed(self.__predict__)(dataset)
            # for dataset in tqdm(splited_data)

        return results

    def train_model(
        self, regr, X_train: np.array, y_train: np.array, sample_weights: list
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
        self,
        regr,
        X_test: np.array,
        y_test: np.array,
        test: np.array,
        method: str = "O1",
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
        test_error = y_pred - y_test

        match method:
            case "O1":  # error_ascend
                index = np.argmin(abs(test_error))
            case "O2":  # error_descend
                index = np.argmax(abs(test_error))
            case "O3":  # bandgap_ascend
                index = np.argmin(y_test)
            case "O4":  # bandgap_descend
                index = np.argmax(y_test)

        return index, test[index], y_pred[index]

    def learning_rate(
        self, error: float, min_error: float, step: float, ranges: list, multiples: list
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

    # def LoadCost(index: int, costs: np.array):

    def iml_learn_train(self, regr, train: list, sample_weights: list) -> Tuple:
        """Imbalance Learning basic training function

        To train machine learning model with sample weights.

        Args:
            regr (model): the basic ML model with hyperparameters.
            train (list): instance indexes which will be used to train the ML model.
            sample_weights (list): weights of training instances.

        Returns:
            Tuple[model, float]: return trained model and prediciton error of last
            (newly added) training instance .

        """
        X = self.X
        y = self.y
        X_train = X[train, :]
        y_train = y[train]
        regr, metrics = self.train_model(regr, X_train, y_train, sample_weights)
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
            print(f"Opt cliff ->", end=" ")

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

    @timer
    # @nb.jit()
    # @delayed
    # @wrap_non_picklable_objects
    def predict_with_iml(self, dataset: np.array) -> np.array:
        """Imbalance Learning main function

        Args:
            dataset (np.array): splitted dataset of instance indexes, including majority
                training, majority validation, majority test, minority training, minority
                validation and minority test sets (totally 6 sets).

        Returns:
            result (np.array): return prediction result for one dataset.

        """
        data = self.data
        X = self.X
        y = self.y
        regr = self.regr
        eta = self.eta
        op_method = self.op_method
        limit = self.limit
        DEBUG = self.DEBUG
        CHECK = self.CHECK

        majority_train_set = dataset[0]
        minority_train_set = dataset[3]
        majority_train_size = dataset[0].shape[0]
        minority_train_size = dataset[3].shape[0]

        train = majority_train_set

        sample_weights = np.ones(len(train), dtype=float)
        coefs = np.ones(minority_train_size, dtype=float)
        order = []

        # if parameters["phar"]:
        # pbar = tqdm(total=minority_train_size)
        # pbar.set_description("SIML (iteration %s)" % n)

        for epoch in range(minority_train_size):
            X_train = X[train, :]
            y_train = y[train]
            regr, _ = self.train_model(regr, X_train, y_train, sample_weights)

            if CHECK:
                best = minority_train_set[epoch]
                best_pred = regr.predict(X)[best]
            else:
                best_index, best, best_pred = self.test_error(
                    regr,
                    X[minority_train_set, :],
                    y[minority_train_set],
                    minority_train_set,
                    op_method,
                )

            y_best = y[best]
            order = np.append(order, str(best))
            error = abs(y_best - best_pred)

            if DEBUG:
                print(f"{best:3}: 1.00({(error-limit):.6f}) ->", end=" ")

            coef = self.iml_learn(train, best, sample_weights, error)
            coefs[epoch] = coef

            if not CHECK:
                sample_weights = np.append(sample_weights, coef)
                train = np.append(train, best)
                minority_train_set = np.delete(minority_train_set, best_index)

        # if parameters["nonOrder"]:
        #     if error == old_error and j == 1:
        #         sys.exit(1)

        #     if parameters["tolerance2"]:
        #         tolerance2 = parameters["tolerance2"]
        #         limit = round(epsilon + tolerance2, 6)

        #     j = 0
        #     while mae > limit:

        #         old_mae = mae

        #         regr = SVR(epsilon=epsilon, kernel="rbf", C=C, gamma=gamma)
        #         (
        #             regr,
        #             train_error,
        #             train_mae,
        #             train_var,
        #             train_rmse,
        #             train_mape,
        #             train_r2,
        #             train_std,
        #         ) = self.train_model(regr, X_train, y_train, sample_weights)
        #         error = abs(y_best - regr.predict(X_best.reshape(1, -1)))[0]
        #         if parameters["wMAE"]:
        #             mae = (np.sum(train_error) + coefs[epoch] * error) / len(train)
        #         else:
        #             mae = (np.sum(train_error) + error) / len(train)

        #         if mae > limit:
        #             if mae == old_mae and j > 0:
        #                 sys.exit(1)
        #             eta = self.learning_rate(error, limit, parameters["eta"], ranges1, multiples1)
        #             C += eta

        #         j += 1

        result = []

        if CHECK:
            keys = index_to_name(data, minority_train_set)
            values = coefs
            result.append(minority_train_set)
            result.append(y[minority_train_set])
            result.append(keys)
            result.append(values)
        else:
            y_pred = regr.predict(X)

            for i in range(6):
                df = dataset[i]
                result.append(index_to_name(data, df))
                result.append(y[df])
                result.append(y_pred[df])

            keys = index_to_name(data, train[majority_train_size:])
            values = coefs
            result.append(keys)
            result.append(values)

            # parameters["C"] = C
            # result.append(list(parameters.keys()))
            # result.append(list(parameters.values()))

        # print(f"------------------------")
        return result

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
            coef (float): return the weight for the best instance.

        """
        X = self.X
        y = self.y
        regr = self.regr
        ranges_arr = self.ranges
        multiples_arr = self.multiples
        delta = self.delta
        eta = self.eta
        wMAE = self.wMAE
        DEBUG = self.DEBUG
        coef = 1.0
        limit = self.limit

        old_coef = coef
        sample_weights = np.append(sample_weights, coef)
        train = np.append(train, best)
        X_train = X[train, :]
        y_train = y[train]
        regr, metrics = self.train_model(regr, X_train, y_train, sample_weights)
        train_error = metrics["error"]

        ranges1 = [round(i * limit, 3) for i in ranges_arr]
        multiples1 = multiples_arr

        step = 0
        old_delta = 0
        IS_FLAT = False

        start_at = time.time()

        while error > limit:
            old_error = error
            old_coef = coef
            delta = self.learning_rate(error, limit, self.delta, ranges1, multiples1)
            old_delta = delta
            coef = round(old_coef + delta, 2)
            sample_weights[len(train) - 1] = coef
            step += 1
            regr, error = self.iml_learn_train(regr, train, sample_weights)

            if DEBUG:
                print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")

            if old_delta == delta and delta <= 1.0 and error > limit:
                IS_FLAT = True
                coef, error, step, delta = self.iml_learn_cliff(
                    train, error, sample_weights, delta, old_coef, step
                )

            # if DEBUG:
            #     print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")
            # if wMAE:
            #     mae = (np.sum(train_error[:-1]) + coef * error) / len(train)
            # else:
            #     mae = (np.sum(train_error[:-1]) + error) / len(train)

            # if not (parameters["nonC"] or parameters["nonOrder"]):
            #     if error == old_error and j == 1:
            #         sys.exit(1)

            #     # if parameters["tolerance2"]:
            #     #     tolerance2 = parameters["tolerance2"]
            #     #     limit = round(epsilon + tolerance2, 6)

            #     j = 0
            #     while mae > limit:

            #         old_mae = mae

            #         regr = SVR(epsilon=epsilon, kernel="rbf", C=C, gamma=gamma)
            #         regr, metrics = train_model(regr, X_train, y_train, sample_weights)
            #         error = train_error[-1]
            #         if wMAE:
            #             mae = (np.sum(train_error) + coef * error) / len(train)
            #         else:
            #             mae = (np.sum(train_error) + error) / len(train)

            #         if mae > limit:
            #             if mae == old_mae and j > 0:
            #                 sys.exit(1)
            #             eta = self.learning_rate(
            #                 error, limit, parameters["eta"], ranges1, multiples1
            #             )
            #             C += eta

            #         j += 1

            # else:
            #     if error == old_error:
            #         sys.exit(1)

        # Trackback
        if IS_FLAT and delta <= self.increments[0]:
            print(f"")
        else:
            coef, error, step = self.iml_learn_track(
                train, error, sample_weights, delta, old_coef, step
            )

        if error <= limit:
            end_at = time.time()
            if wMAE:
                mae = (np.sum(train_error[:-1]) + coef * error) / len(train)
            else:
                mae = (np.sum(train_error[:-1]) + error) / len(train)
            sample_weights[len(train) - 1] = coef
            if DEBUG:
                print(
                    f"END (coef: {coef:.2f}, "
                    f"error: {(error - limit):.6f}, step: {step}, "
                    f"used time: {(end_at - start_at):.3f}s)"
                )
            # if parameters["costr"]:
            #     cost_result = []
            #     cost_result.append(best)
            #     cost_result.append(step)
            #     cost_result.append(coef)
            #     cost_results.append(cost_result)
            # if parameters["phar"]:
            #     pbar.update(1)

        return coef

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
            print(f"Trackback ->", end=" ")
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

        if DEBUG:
            if coef != (coef_bak):
                print(f"{coef:.2f}({(error - limit):.6f}) ->", end=" ")

        return coef, error, step

    @timer
    def iml(self) -> list:
        """Imbalance Learning control function

        To handle the computing mode: single core or parallel with multiple cores.

        Returns:
            results (list): return prediction results for splited datasets.

        """
        splited_data = self.splited_data
        n_jobs = self.n_jobs
        iterations = self.iterations
        results = []

        # match parameters["wb"]:
        #     case True:
        #         if parameters["parallel"]:
        #             results = Parallel(n_jobs=n_jobs)(
        #                 delayed(self.iml_check)(n)
        #                 for n in tqdm(range(iterations))
        #             )
        #         else:
        #             for n in range(1, len(splited_data) + 1, 1):
        #                 result = self.iml_check(n)
        #                 results.append(result)
        #     case _:
        #         if parameters["costr"]:
        #             result = self.predict_with_iml(1)
        #             results.append(result)
        #         else:
        #             if parameters["parallel"]:
        #                 results = Parallel(n_jobs=n_jobs)(
        #                     delayed(self.predict_with_iml)(n)
        #                     for n in tqdm(range(iterations))
        #                 )
        #             else:
        #                 for n in range(1, len(splited_data) + 1, 1):
        #                     result = self.predict_with_iml(n)
        #                     results.append(result)

        # for n in range(1, iterations + 1, 1):
        #   result = self.predict_with_iml(n)

        # predict_with_iml = self.predict_with_iml

        if n_jobs == 1:
            results = []
            pbar = tqdm(total=iterations)
            pbar.set_description("SIML")
            for dataset in splited_data:
                result = self.predict_with_iml(dataset)
                results.append(result)
                pbar.update(1)
        else:
            results = Parallel(n_jobs, verbose=3)(
                delayed(self.predict_with_iml)(splited_data[i])
                for i in tqdm(range(iterations))
                # delayed(self.predict_with_iml)(dataset)
                # for dataset in splited_data
            )

        return results
