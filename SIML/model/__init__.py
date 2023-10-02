"""Initialization for model module"""

from ._base import BaseModel
from .basis1 import Basis1Model
from .basis2 import Basis2Model
from .iml import IML, IML2, learning_rate, test_error, train_model

__all__ = [
    "BaseModel",
    "Basis1Model",
    "Basis2Model",
    "IML",
    "IML2",
    "train_model",
    "learning_rate",
    "test_error",
]
