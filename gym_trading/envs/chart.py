"""
This module provides classes to define charts with dates and values.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

import pandas as pd
from pandas import DataFrame, Series


class Chart(ABC):
    """Chart with dates and values."""

    @abstractmethod
    def data_at(self, date: datetime) -> Optional[DataFrame]:
        """
        Returns the value at the given date.

        Args:
            date (datetime): The date for which data is requested.

        Returns:
            Optional[DataFrame]: The data at the given date or None if not found.
        """

    @abstractmethod
    def add(self, data: DataFrame):
        """
        Adds the given data frame to the chart.

        Args:
            data (DataFrame): The data frame to add to the chart.
        """

    @abstractmethod
    def window(self, start_date: datetime, end_date: datetime) -> DataFrame:
        """
        Returns a data frame in the given range.

        Args:
            start_date (datetime): The start date of the range.
            end_date (datetime): The end date of the range.

        Returns:
            DataFrame: The data frame containing data within the specified range.
        """

    @abstractmethod
    def timestamps(self) -> List[datetime]:
        """
        Returns a list of timestamps.

        Returns:
            List[datetime]: A list of timestamps.
        """


class DataChart(Chart):
    """Chart with dates and values."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        timestamp_column_name: str,
    ):
        """
        Initialize a DataChart instance.

        Args:
            dataset (pd.DataFrame): The dataset containing timestamped data.
            timestamp_column_name (str): The name of the timestamp column in the dataset.
        """
        self.dataset = dataset
        self.timestamp_column_name = timestamp_column_name

        self.dataset[self.timestamp_column_name] = pd.to_datetime(
            self.dataset[self.timestamp_column_name]
        )

    def data_at(self, date: datetime) -> Optional[pd.DataFrame]:
        """
        Returns data at the given date.

        Args:
            date (datetime): The date for which data is requested.

        Returns:
            Optional[pd.DataFrame]: The data at the given date or None if not found.
        """
        filtered_data = self.dataset[self.dataset[self.timestamp_column_name] == date]
        return filtered_data.reset_index(drop=True) if not filtered_data.empty else None

    def add(self, data: pd.DataFrame):
        """
        Adds the given data frame to the chart.

        Args:
            data (pd.DataFrame): The data frame to add to the chart.
        """
        self.dataset = pd.concat([self.dataset, data], ignore_index=True)

    def window(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Returns a data frame in the given date range.

        Args:
            start_date (datetime): The start date of the range.
            end_date (datetime): The end date of the range.

        Returns:
            pd.DataFrame: The data frame containing data within the specified range.
        """
        filtered_data = self.dataset[
            (self.dataset[self.timestamp_column_name] >= start_date)
            & (self.dataset[self.timestamp_column_name] <= end_date)
        ]
        sorted_data = filtered_data.sort_values(
            by=self.timestamp_column_name, ascending=True
        )
        return sorted_data

    def timestamps(self) -> List[datetime]:
        """
        Returns a list of timestamps.

        Returns:
            List[datetime]: A list of timestamps.
        """
        return self.dataset[self.timestamp_column_name].sort_values().tolist()


class AssetDataChart(DataChart):
    """Asset chart with dates and values."""

    def __init__(
        self, dataset: pd.DataFrame, timestamp_column_name: str, price_column_name: str
    ):
        """
        Initialize an AssetDataChart instance.

        Args:
            dataset (pd.DataFrame): The dataset containing timestamped data.
            timestamp_column_name (str): The name of the timestamp column in the dataset.
            price_column_name (str): The name of the price column in the dataset.
        """
        super().__init__(dataset, timestamp_column_name)
        self.price_column_name = price_column_name

    def price_at(self, date: datetime):
        """
        Returns the price at the given date.

        Args:
            date (datetime): The date for which the price is requested.

        Returns:
            float: The price at the given date.
        """
        return self.data_at(date)[self.price_column_name].iloc[0]

    def prices(self) -> Series:
        """
        Returns a Series of prices.

        Returns:
            Series: A Series containing prices.
        """
        return self.dataset[self.price_column_name]
