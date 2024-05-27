"""Dataclasses for analysis"""

from dataclasses import dataclass


@dataclass
class ResData:
    """Result data

    Args:
        y (list): all y values.
        y_pred (list): all y prediction values.
        y_majority (list): all y majority values.
        y_majority_pred (list): all y majority prediction values.
        y_minority (list): all y minority values.
        y_minority_pred (list): all y minority prediction values.
    """

    y: list
    y_pred: list
    y_majority: list
    y_majority_pred: list
    y_minority: list
    y_minority_pred: list
