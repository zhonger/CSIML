"""Dataclasses for cross validation module"""
from dataclasses import dataclass

import numpy as np


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
        random_state2 (int): the random seed for oversampling or undersampling.
    """

    majority_pro: float
    minority_pro: float
    majority_test_num: int
    random_state: int
    random_state2: int
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
                "Validation" and "Test". For example, {"Total": 20, "Train": "10", 
                "Validation": 8, "Test": 2}.

        Raises:
            AttributeError: when setting a inavailable attribute.
        """
        for key, value in values.items():
            try:
                getattr(self, key.lower())
                setattr(self, key.lower(), value)
            except AttributeError as e:
                raise AttributeError(f"{key} is not available.") from e
