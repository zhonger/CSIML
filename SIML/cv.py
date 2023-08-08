import math
from dataclasses import dataclass
from random import sample, seed
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import ShuffleSplit


@dataclass
class Setting:
    """A datacalss for splitting setting

    Args:
        majority_pro (float): the proportion for majority.
        minority_pro (float): the proportion for majority.
        majority_test_num (int): majority test size (only for ``basis1`` cross
            validation method).
        majority_set (np.array): the indexes of majority set.
        minority_set (np.array): the indexes of minority set.
        random_state (int): the random seed for random sampling.

    """

    majority_pro: float
    minority_pro: float
    majority_test_num: int
    random_state: int
    majority_set: np.array
    minority_set: np.array


@dataclass
class DataSize:
    """A dataclass for data size description

    Args:
        total (int): the number of all data.
        train (int): the number of training data. Defaults to 0.
        validation (int): the number of validation data. Defaults to 0.
        test (int): the number of test data. Defaults to 0.

    """

    total: int
    train: int = 0
    validation: int = 0
    test: int = 0

    def __str__(self) -> str:
        """Print data size"""
        msg = (
            f"Total: {self.total}, Train: {self.train}, "
            f"Validation: {self.validation}, Test: {self.test}"
        )
        return msg

    def set(self, values: dict) -> None:
        """Set values for specified attributes

        Args:
            values (dict): a dict for attributes, supporting "Total", "Train",
                "Validation" and "Test".

        Raises:
            AttributeError: when setting a inavailable attribute.

        """
        for key, value in values.items():
            try:
                getattr(self, key.lower())
                setattr(self, key.lower(), value)
            except AttributeError:
                raise AttributeError(f"{key} is not available.")


class DatasetSize:
    """A class for dataset size description

    Args:
        data (pd.DataFrame): the initial dataset, which should include
            "Materials", "Experimental" columns.
        cv_method (str, optional): cross validation method, supporting
            "pls", "jpcl", "siml", "basis1" and "op". Defaults to "siml".
        sampling_method (str, optional): resampling method, supporting
            "oversampling" and "undersampling". Defaults to None.
        threshold (float, optional): for splitting majority and minority.
            Defaults to 5.0 (for bandgaps).

    Note:
        The iteration of supported ``cv_method``:
            - ``siml``: (4:1)4:1 (5-fold CV, 5x5x5x5=625 iterations)
            - ``pls``:  1,2,3,4,5,6,7,... (10-fold CV, 10 iterations)
            - ``jpcl``: 1,2,3,4,5,1,2,3,4,5,... (5-fold CV, 5 iterations)
            - ``basis1``: 24 (test), 4:1(Train vs Validation) (5-fold CV, 5 iterations)

    Attributes:
        majority (DataSize): including 'total', 'train', 'validation', 'test'.
        minority (DataSize): including 'total', 'train', 'validation', 'test'.
        splited_data (numpy.array): the splited indexes of data in cross validation.
        iterations (int): the iteration of cross validation.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        cv_method: str = "siml",
        sampling_method: str = None,
        threshold: float = 5.0,
    ) -> None:
        """Initial DatasetSize class"""
        self.data = data
        self.size = data.shape[0]
        self.cv_method = cv_method
        self.sampling_method = sampling_method
        majority_total = len(data[data["Experimental"] < threshold])
        minority_total = self.size - majority_total
        self.majority = DataSize(majority_total)
        self.minority = DataSize(minority_total)
        self.splited_data = []
        self.iterations = 0

    def __str__(self) -> str:
        """Print dataset size including majority and minority"""
        msg = f"Majority: {self.majority.__dict__}\nMiniroty: {self.minority.__dict__}"
        return msg

    def set(self, majority: dict, minority: dict) -> None:
        """Set majority and minority attributes

        Args:
            majority (dict): supporting "Total", "Train", "Validation" and "Test".
            minority (dict): supporting "Total", "Train", "Validation" and "Test".

        """
        self.majority.set(majority)
        self.minority.set(minority)

    def k_fold_size(self, fold: int = 10) -> Tuple[dict, dict]:
        """Calculate the size when using k fold cross validation method

        Args:
            fold (int, optional): 10 or 5 (for pls and jpcl CV). Defaults to 10.

        Returns:
            Tuple[dict, dict]: return majority and minority dicts.

        """
        k = 1 / fold

        majority_total = self.majority.total
        majority_validation = math.ceil(majority_total * k)
        majority_train = majority_total - majority_validation
        minority_total = self.minority.total
        minority_validation = math.ceil(minority_total * k)
        minority_train = minority_total - minority_validation
        majority_test = minority_test = 0

        majority = {
            "Total": majority_total,
            "Train": majority_train,
            "Validation": majority_validation,
            "Test": majority_test,
        }
        minority = {
            "Total": minority_total,
            "Train": minority_train,
            "Validation": minority_validation,
            "Test": minority_test,
        }

        return majority, minority

    def cal_size(
        self,
        majority_pro: float = 0.2,
        minority_pro: float = 0.2,
        majority_test_num: int = 24,
    ) -> None:
        """Calculate dataset sizes for training, validation and test set of
            majority and minority

        Args:
            majority_pro (float, optional): the proportion for majority.
                Defaults to 0.2.
            minority_pro (float, optional): the proportion for minority.
                Defaults to 0.2.
            majority_test_num (int, optional): majority test size (only for
                ``basis1`` cross validation method). Defaults to 24.

        """
        cv_method = self.cv_method
        sampling_method = self.sampling_method
        majority_total = self.majority.total
        minority_total = self.minority.total

        if cv_method == "pls":
            majority, minority = self.k_fold_size()
        elif cv_method == "jpcl":
            majority, minority = self.k_fold_size(5)
        else:
            if cv_method == "basis1":
                majority_test = majority_test_num
                majority_rest = majority_total - majority_test_num
                majority_validation = math.ceil(majority_rest * majority_pro)
                majority_train = majority_rest - majority_validation
                minority_train = minority_validation = 0
                minority_test = minority_total
            else:
                # Other cv_methods (siml, op), default 'siml'
                if cv_method == "op":
                    majority_pro = minority_pro = 0.1

                majority_test = math.ceil(majority_total * majority_pro)
                majority_rest = majority_total - majority_test
                majority_validation = math.ceil(majority_rest * majority_pro)
                majority_train = majority_rest - majority_validation
                minority_test = math.ceil(minority_total * minority_pro)
                minority_rest = minority_total - minority_test
                minority_validation = math.ceil(minority_rest * minority_pro)
                minority_train = minority_rest - minority_validation

            majority = {
                "Total": majority_total,
                "Train": majority_train,
                "Validation": majority_validation,
                "Test": majority_test,
            }
            minority = {
                "Total": minority_total,
                "Train": minority_train,
                "Validation": minority_validation,
                "Test": minority_test,
            }

        if sampling_method == "oversampling":
            minority["Total"] = (
                minority["Total"] - minority["Train"] + majority["Train"]
            )
            minority["Train"] = majority["Train"]
        elif sampling_method == "undersampling":
            majority["Total"] = (
                majority["Total"] - majority["Train"] + minority["Train"]
            )
            majority["Train"] = minority["Train"]

        self.set(majority, minority)

    def over_sampling(
        self, X: pd.DataFrame, y: pd.DataFrame, random_state: int = 3
    ) -> Tuple[list, list]:
        """Using over sampling method to generate more instances for minority

        Args:
            X (pd.DataFrame): the features
            y (pd.DataFrame): the property
            random_state (int, optional): random seed. Defaults to 3.

        Returns:
            Tuple[list, list]: return splitted X and y.

        """
        ros = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        return X_resampled, y_resampled

    def under_sampling(
        self, X: pd.DataFrame, y: pd.DataFrame, random_state: int = 3
    ) -> Tuple[list, list]:
        """Using under sampling method to delete instances for majority

        Args:
            X (pd.DataFrame): the features.
            y (pd.DataFrame): the property.
            random_state (int, optional): random seed. Defaults to 3.

        Returns:
            Tuple[list, list]: return splitted X and y.

        """
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    def resampling(
        self,
        majority_train: list,
        minority_train: list,
        random_state: int = 3,
        threshold: float = 5.0,
    ) -> Tuple[list, list]:
        """Help function for over sampling and under sampling

        Args:
            majority_train (list): the majority training index set.
            minority_train (list): the minority training index set.
            random_state (int, optional): random seed. Defaults to 3.

        Returns:
            Tuple[list, list]: return majority and minority training sets.

        """
        data = self.data
        sampling_method = self.sampling_method

        if sampling_method == "oversampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train, minority_train)
            c = y[train] <= threshold
            train = train[:, np.newaxis]
            train_resample, c = self.over_sampling(train, c, random_state=random_state)
            train_resample = train_resample.flatten()
            minority_train = train_resample[len(majority_train) :]
        elif sampling_method == "undersampling":
            y = data.iloc[:, 1].values
            train = np.append(majority_train, minority_train)
            c = y[train] <= threshold
            train = train[:, np.newaxis]
            train_resample, c = self.under_sampling(train, c, random_state=random_state)
            train_resample = train_resample.flatten()
            majority_train = train_resample[len(minority_train) :]

        return majority_train, minority_train

    def split_data(
        self,
        majority_pro: float = 0.2,
        minority_pro: float = 0.2,
        random_state: int = 3,
        threshold: float = 5.0,
        majority_test_num: int = 24,
    ) -> None:
        """Split dataset into training, validation and test sets

        Args:
            majority_pro (float, optional): the proportion for majority.
                Defaults to 0.2.
            minority_pro (float, optional): the proportion for minority.
                Defaults to 0.2.
            random_state (int, optional): random seed. Defaults to 3.
            threshold (float, optional): the threshold to distinguish majority
                and minority. Defaults to 5.0 (for bandgap).
            majority_test_num (int, optional): majority test size (only for
                ``basis1`` cross validation method). Defaults to 24.

        """
        data = self.data
        data_size = self.size
        cv_method = self.cv_method
        majority_size = len(data[data["Experimental"] < threshold])
        minority_size = data_size - majority_size
        majority_set = np.arange(majority_size)
        minority_set = np.arange(data_size)[-minority_size:]
        settings = Setting(
            majority_pro,
            minority_pro,
            majority_test_num,
            random_state,
            majority_set,
            minority_set,
        )

        match cv_method:
            case "siml":
                self.split_data_siml(settings)
            case "pls":
                self.split_data_kfoldcv(settings)
            case "jpcl":
                self.split_data_kfoldcv(settings, 5)
            case "basis1":
                self.split_data_basis(settings)
            case "op":
                self.split_data_kfoldcv(settings)

        self.iterations = len(self.splited_data)

    def split_data_siml(self, settings: Setting) -> None:
        """Split dataset according to siml cross validation method

        Args:
            settings (Setting): for splitting.

        """
        majority_pro = settings.majority_pro
        minority_pro = settings.minority_pro
        random_state = settings.random_state
        majority = settings.majority_set
        minority = settings.minority_set

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
        splited_data = []

        for min_r, min_t in ss2.split(minority):
            for min_tr, min_v in ss2.split(min_r):
                for maj_r, maj_t in ss1.split(majority):
                    for maj_tr, maj_v in ss1.split(maj_r):
                        cv_data = []

                        majority_train = majority[maj_r[maj_tr]]
                        majority_validation = majority[maj_r[maj_v]]
                        majority_test = majority[maj_t]
                        minority_train = minority[min_r[min_tr]]
                        minority_validation = minority[min_r[min_v]]
                        minority_test = minority[min_t]

                        majority_train, minority_train = self.resampling(
                            majority_train, minority_train, random_state
                        )

                        cv_data.append(majority_train)
                        cv_data.append(majority_validation)
                        cv_data.append(majority_test)
                        cv_data.append(minority_train)
                        cv_data.append(minority_validation)
                        cv_data.append(minority_test)

                        splited_data.append(cv_data)

        self.splited_data = splited_data

    def split_data_basis(self, settings: Setting) -> None:
        """Split dataset according to basis1 cross validation method

        Args:
            settings (Setting): for splitting.

        """
        majority_pro = settings.majority_pro
        random_state = settings.random_state
        majority_set = settings.majority_set
        minority_set = settings.minority_set
        majority_test_num = settings.majority_test_num

        ss1 = ShuffleSplit(
            n_splits=int(1 / majority_pro),
            test_size=majority_pro,
            random_state=random_state,
        )

        splited_data = []

        majority_test = sample(list(majority_set), majority_test_num)
        majority_set = list(set(majority_set) - set(majority_test))
        minority_test = minority_set

        for majority_train, majority_validation in ss1.split(majority_set):
            cv_data = []

            cv_data.append(majority_train)
            cv_data.append(majority_validation)
            cv_data.append(np.array(majority_test))
            cv_data.append(minority_test)

            splited_data.append(cv_data)

        self.splited_data = splited_data

    def split_data_kfoldcv(self, settings: Setting, fold: int = 10) -> None:
        """Split dataset according to k-fold or op cross validation method

        Args:
            settings (Setting): for splitting.
            fold (int, optional): the fold for cross validation. Defaults to 10.

        """
        random_state = settings.random_state
        majority = settings.majority_set
        minority = settings.minority_set
        cv_method = self.cv_method
        splited_data = []

        if cv_method == "op":
            # Obtain test_set for majority and minority
            seed(a=random_state)
            majority_size_test = self.majority.test
            minority_size_test = self.minority.test
            majority_test = sample(list(majority), majority_size_test)
            majority = np.array(list(set(majority) - set(majority_test)))
            minority_test = sample(list(minority), minority_size_test)
            minority = np.array(list(set(minority) - set(minority_test)))
            majority_test = np.array(majority_test)
            minority_test = np.array(minority_test)

        for i in range(fold):
            cv_data = []

            if cv_method == "op":
                majority_indexes = np.arange(majority.shape[0])
                majority_validation_indexes = majority_indexes % fold == i
                majority_validation = majority[majority_validation_indexes]
                majority_train = np.delete(majority, majority_validation_indexes)
            else:
                majority_validation = majority[majority % fold == i]
                majority_test = majority_validation
                majority_train = np.delete(majority, majority_validation)

            minority_indexes = np.arange(minority.shape[0])
            minority_validation_indexes = minority_indexes % fold == i
            minority_validation = minority[minority_validation_indexes]

            if cv_method != "op":
                minority_test = minority_validation

            minority_train = np.delete(minority, minority_validation_indexes)

            majority_train, minority_train = self.resampling(
                majority_train, minority_train, random_state
            )

            cv_data.append(majority_train)
            cv_data.append(majority_validation)
            cv_data.append(majority_test)
            cv_data.append(minority_train)
            cv_data.append(minority_validation)
            cv_data.append(minority_test)

            splited_data.append(cv_data)

        self.splited_data = splited_data
