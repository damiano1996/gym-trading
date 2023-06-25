"""
This module provides the implementation of an exchange.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from gym_trading.envs.chart import ChartDataFrame


class Exchange(ABC):
    """Abstract base class for exchange implementations."""

    @abstractmethod
    def get_equity_history(self) -> ChartDataFrame:
        """
        Get the equity history of the exchange.

        Returns:
            ChartDataFrame: The equity history as a ChartDataFrame object.
        """

    @abstractmethod
    def buy_all(self, price: float, date: datetime) -> bool:
        """
        Buy all available equity at the specified price and date.

        Args:
            price (float): The price at which to buy.
            date (datetime): The date of the buy order.

        Returns:
            bool: True if the buy order was successful, False otherwise.
        """

    @abstractmethod
    def sell_all(self, price: float, date: datetime) -> bool:
        """
        Sell all held equity at the specified price and date.

        Args:
            price (float): The price at which to sell.
            date (datetime): The date of the sell order.

        Returns:
            bool: True if the sell order was successful, False otherwise.
        """

    @abstractmethod
    def hold(self, price: float, date: datetime):
        """
        Hold equity at the specified price and date.

        Args:
            price (float): The price at which to hold.
            date (datetime): The date of the hold order.
        """

    @abstractmethod
    def reset(self):
        """
        Reset the exchange state.
        """


class BaseExchange(Exchange):
    """A base exchange class for buying and selling assets."""

    def __init__(
            self,
            init_amount=100.0,
            buy_fee=0.25,
            sell_fee=0.25
    ):
        """
        Initialize the BaseExchange object.

        Args:
            init_amount (float, optional): The initial amount of the asset. Defaults to 100.0.
            buy_fee (float, optional): The buy fee percentage. Defaults to 0.25.
            sell_fee (float, optional): The sell fee percentage. Defaults to 0.25.
        """
        self.init_amount = init_amount

        self.current_amount = self.init_amount
        self.is_reference_asset = True

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        self.equities = ChartDataFrame({'DATE': [], 'VALUE': []})

    def get_equity_history(self) -> ChartDataFrame:
        """
        Get the equity history.

        Returns:
            ChartDataFrame: The equity history.
        """
        return self.equities

    def buy_all(self, price: float, date: datetime) -> bool:
        """
        Buy all the assets.

        Args:
            price (float): The price of the asset.
            date (datetime): The date of the transaction.

        Returns:
            bool: True if the buy operation is successful, False otherwise.
        """
        if self.is_reference_asset:
            # saving equity considering fees
            self.equities.add_data(date, self.add_percentage(
                self.current_amount, -self.buy_fee))
            # converting to target asset
            self.current_amount = self.add_percentage(
                self.current_amount / price, -self.buy_fee)
            self.is_reference_asset = False
            return True
        return False

    def sell_all(self, price: float, date: datetime) -> bool:
        """
        Sell all the assets.

        Args:
            price (float): The price of the asset.
            date (datetime): The date of the transaction.

        Returns:
            bool: True if the sell operation is successful, False otherwise.
        """
        if not self.is_reference_asset:
            self.current_amount = self.add_percentage(
                self.current_amount * price, -self.sell_fee)
            self.is_reference_asset = True
            # saving current amount in reference asset
            self.equities.add_data(date, self.current_amount)
            return True
        return False

    def hold(self, price: float, date: datetime):
        """
        Hold the current amount of assets.

        Args:
            price (float): The price of the asset.
            date (datetime): The date of the transaction.
        """
        self.equities.add_data(
            date,
            self.equities.values[-1] if len(self.equities.values) > 0 else self.init_amount
        )

    def reset(self):
        """Reset the exchange to its initial state."""
        self.current_amount = self.init_amount
        self.is_reference_asset = True

        self.equities = ChartDataFrame({'DATE': [], 'VALUE': []})

    @staticmethod
    def add_percentage(value, percentage):
        """
        Add a percentage to a value.

        Args:
            value (float): The value to which the percentage is added.
            percentage (float): The percentage to be added.

        Returns:
            float: The value with the added percentage.
        """
        add = percentage * np.abs(value) / 100.
        return value + add
