"""Periodic table helper module"""
import os

import pandas as pd

from ._dataclass import Element


class PT:
    """Periodic table helper

    Attributes:
        pt (pd.DataFrame): periodic table object, including some basic information.
    """

    def __init__(self) -> None:
        self.pt = pd.read_csv(f"{os.path.dirname(__file__)}/periodictable.csv", header=None)

    def get_element_symbol(self, number: int) -> str:
        """Get element symbol by element number

        Args:
            number (int): the element number.

        Returns:
            str: the element symbol.
        """
        symbol = self.pt.iloc[number - 1, 1]
        return symbol

    def get_element_number(self, symbol: str) -> int:
        """Get element number by element symbol

        Args:
            symbol (str): the element symbol.

        Returns:
            int: the element number.
        """
        number = self.pt[self.pt[1] == symbol].iloc[0, 0]
        return number

    def get_element(self, **kws) -> Element:
        """Get element object

        It will use symbol to obtain the information for the element by default. If no
        symbol, the element number will be requred.

        Returns:
            Element: the element object.
        """
        if "symbol" in kws:
            number = self.get_element_number(kws.get("symbol"))
        elif "number" in kws:
            number = kws.get("number")
        else:
            raise ValueError("Please give 'symbol' or 'number'!")
        element = Element(
            number,
            self.pt.iloc[number - 1, 1],
            int(self.pt.iloc[number - 1, 4]),
            int(self.pt.iloc[number - 1, 5]),
        )
        return element
