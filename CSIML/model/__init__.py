"""Initialization for model module"""

from ._base import BaseModel
from .basis1 import Basis1Model
from .basis2 import Basis2Model
from .iml import IML, test_error, train_model

__all__ = [
    "BaseModel",
    "Basis1Model",
    "Basis2Model",
    "IML",
    "train_model",
    "test_error",
]
