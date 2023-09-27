"""
This module provides classes to define charts with dates and values.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

import pandas as pd
from pandas import DataFrame, Series


class Chart(ABC):

    @abstractmethod
    def data_at(self, date: datetime) -> Optional[DataFrame]:
        """
            Returns the value at the given date.
        """

    @abstractmethod
    def add(self, data: DataFrame):
        """
            Adds the given data frame to the chart.
        """

    @abstractmethod
    def window(self, start_date: datetime, end_date: datetime) -> DataFrame:
        """
            Returns a data frame in the given range.
        """

    @abstractmethod
    def timestamps(self) -> List[datetime]:
        """
            Returns a list of timestamps.
        """


class DataChart(Chart):

    def __init__(
            self,
            dataset: pd.DataFrame,
            timestamp_column_name: str,
    ):
        self.dataset = dataset
        self.timestamp_column_name = timestamp_column_name

        self.dataset[self.timestamp_column_name] = pd.to_datetime(self.dataset[self.timestamp_column_name])

    def data_at(self, date: datetime) -> Optional[pd.DataFrame]:
        filtered_data = self.dataset[self.dataset[self.timestamp_column_name] == date]
        return filtered_data.reset_index(drop=True) if not filtered_data.empty else None

    def add(self, data: pd.DataFrame):
        self.dataset = pd.concat([self.dataset, data], ignore_index=True)

    def window(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        filtered_data = self.dataset[
            (self.dataset[self.timestamp_column_name] >= start_date) &
            (self.dataset[self.timestamp_column_name] <= end_date)
            ]
        sorted_data = filtered_data.sort_values(by=self.timestamp_column_name, ascending=True)
        return sorted_data

    def timestamps(self) -> List[datetime]:
        return self.dataset[self.timestamp_column_name].sort_values().tolist()


class AssetDataChart(DataChart):

    def __init__(self, dataset: pd.DataFrame, timestamp_column_name: str, price_column_name: str):
        super().__init__(dataset, timestamp_column_name)
        self.price_column_name = price_column_name

    def price_at(self, date: datetime):
        return self.data_at(date)[self.price_column_name].iloc[0]

    def prices(self) -> Series:
        return self.dataset[self.price_column_name]
