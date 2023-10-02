"""Customized dataclasses for plot module"""
from dataclasses import dataclass

from matplotlib.colors import Colormap, Normalize


@dataclass
class EMetric:
    """A dataclass for error metrics

    Args:
        c1 (int): the number of acceptable error.
        c2 (int): the number of barely acceptable error.
        c3 (int): the number of not acceptable error.
        cp1 (float): the proportion of acceptable error in all errors.
        cp2 (float): the proportion of barely acceptable error in all errors.
        cp3 (float): the proportion of not acceptable error in all errors.
    """

    c1: int
    c2: int
    c3: int
    cp1: float
    cp2: float
    cp3: float


@dataclass
class Element:
    """A dataclass for elements in periodic table

    Args:
        number (int): the element number.
        symbol (str): the element symbol.
        group (int): the element group.
        period (int): the element period.
    """

    number: int
    symbol: str
    group: int
    period: int


@dataclass
class ElementC(Element):
    count: int


@dataclass
class Cells:
    cell_length: float
    cell_edge_width: float
    cell_gap: float
    my_cmap: Colormap
    norm: Normalize
