"""
This module provides the implementation of a data loader.
"""

from abc import abstractmethod, ABC
from typing import List, Dict

import pandas as pd

from gym_trading.envs.chart import AssetDataChart


class AssetChartDataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> Dict[str, AssetDataChart]:
        """Load data and return a Chart object."""


class ListAssetChartDataLoader(AssetChartDataLoader):
    """Data loader for list-based data."""

    def __init__(self, dates: List, prices: List, asset_name: str = "NA"):
        """
        Initialize the ListDataLoader.

        Args:
            dates (List): The list of dates.
            prices (List): The list of corresponding values.
        """
        self.dates = dates
        self.prices = prices

        self.asset_name = asset_name

    def load(self) -> Dict[str, AssetDataChart]:
        data = pd.DataFrame({"Date": self.dates, "Price": self.prices})

        return {
            self.asset_name: AssetDataChart(
                data, timestamp_column_name="Date", price_column_name="Price"
            )
        }


class PandasAssetChartDataLoader(AssetChartDataLoader):
    """Data loader for multiple datasets."""

    def __init__(
        self,
        datasets: Dict[str, pd.DataFrame],
        timestamp_column_name: str,
        price_column_name: str,
    ):
        self.datasets = datasets
        self.timestamp_column_name = timestamp_column_name
        self.price_column_name = price_column_name

    def load(self) -> Dict[str, AssetDataChart]:
        charts = {}
        for name, dataset in self.datasets.items():
            charts[name] = AssetDataChart(
                dataset, self.timestamp_column_name, self.price_column_name
            )

        return charts
