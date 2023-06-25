"""
This module provides the implementation of a data loader.
"""

from abc import abstractmethod, ABC
from typing import List

import pandas as pd

from gym_trading.envs.chart import ChartDataFrame


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> ChartDataFrame:
        """Load data and return a ChartDataFrame object."""


class CsvDataLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, filename):
        """
        Initialize the CsvDataLoader.

        Args:
            filename (str): The path to the CSV file.
        """
        self.filename = filename

    def load(self) -> ChartDataFrame:
        """
        Load data from a CSV file.

        Returns:
            ChartDataFrame: The loaded data as a ChartDataFrame object.
        """
        return pd.read_csv(self.filename)


class ListDataLoader(DataLoader):
    """Data loader for list-based data."""

    def __init__(self, dates: List, values: List):
        """
        Initialize the ListDataLoader.

        Args:
            dates (List): The list of dates.
            values (List): The list of corresponding values.
        """
        self.dates = dates
        self.values = values

    def load(self) -> ChartDataFrame:
        """
        Load data from lists.

        Returns:
            ChartDataFrame: The loaded data as a ChartDataFrame object.
        """
        return ChartDataFrame(
            {
                'DATE': self.dates,
                'VALUE': self.values
            }
        )
