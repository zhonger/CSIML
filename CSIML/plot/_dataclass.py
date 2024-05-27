"""Customized dataclasses for plot module"""

from dataclasses import dataclass, field

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
    """Element dataclass with count

    Args:
        count (int): the count number for the element.

    Other argumemnts please refer to :class:`Element`.
    """

    count: int


@dataclass
class Cells:
    """Cell setting in the periodic table

    Args:
        cell_length (float): the length of the cell.
        cell_edge_width (float): the edge width of the cell.
        cell_gap (float): the gap with the neighbor cell.
        my_cmap (Colormap): the colormap.
        norm (Normalize): the normalization for the colormap.
    """

    cell_length: float
    cell_edge_width: float
    cell_gap: float
    my_cmap: Colormap
    norm: Normalize


@dataclass
class Metric:
    """Metric

    Args:
        error (float, optional): RMSE or MSE value. Defaults to None.
        std (float, optional): the standard deviation for RMSE or MSE value. Defaults to
            None.
    """

    error: list = field(default_factory=list)
    std: list = field(default_factory=list)


@dataclass
class Metrics:
    """Metrics from total, majority and minority set

    Args:
        total (Metric): error and std value for total set.
        maj (Metric): error and std value for majority set.
        min (Metric): error and std value for minority set.
    """

    total: Metric = Metric([], [])
    maj: Metric = Metric([], [])
    min: Metric = Metric([], [])
