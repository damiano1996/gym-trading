"""
This module provides the implementation of an exchange.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import List, Tuple

import numpy as np
import pandas as pd

from gym_trading.envs.data_loader import AssetChartDataLoader


class Exchange(ABC):
    """Abstract base class for exchange implementations."""

    @abstractmethod
    def market_buy(self, asset: str, amount: Decimal, date: datetime) -> bool:
        """Buys the given amount. (amount in reference currency)"""

    @abstractmethod
    def market_sell(self, asset: str, amount: Decimal, date: datetime) -> bool:
        """Buys the given amount.  (amount in reference currency)"""

    @abstractmethod
    def update(self, date: datetime):
        """To update the internal state at the end of a trading day"""

    @abstractmethod
    def equities(self) -> Tuple[List[datetime], List[Decimal]]:
        """Returns the equity at the given date."""

    @abstractmethod
    def budget_distribution(self, date: datetime) -> np.ndarray:
        """Returns the budget distribution in percentage."""

    @abstractmethod
    def reset(self):
        """Resets the exchange status."""


class BaseExchange(Exchange):
    """A base exchange class for buying and selling assets."""

    def __init__(
        self,
        data_loader: AssetChartDataLoader,
        init_liquidity: Decimal = Decimal("100.0"),
        buy_fee: Decimal = Decimal("0.1"),
        sell_fee: Decimal = Decimal("0.1"),
    ):
        self.charts = data_loader.load()

        self.init_liquidity = init_liquidity
        self.liquidity = self.init_liquidity
        self.wallet = {asset: Decimal("0.0") for asset in self.charts.keys()}

        self.equity_history = pd.DataFrame({"Date": [], "Equity": []})

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

    def reset(self):
        self.liquidity = self.init_liquidity
        self.wallet = {asset: Decimal("0.0") for asset in self.charts.keys()}
        self.equity_history = pd.DataFrame({"Date": [], "Equity": []})

    def market_buy(self, asset: str, amount: Decimal, date: datetime):
        if amount < 0:
            return False

        if amount > self.liquidity:
            return False

        if asset not in list(self.charts.keys()):
            return False

        price = Decimal(str(self.charts[asset].price_at(date)))
        asset_amount = amount / price

        self.wallet[asset] += self._add_percentage(asset_amount, -self.buy_fee)
        self.liquidity -= amount

        return True

    def market_sell(self, asset: str, amount: Decimal, date: datetime):
        if amount < Decimal("0"):
            return False

        if asset not in list(self.charts.keys()):
            return False

        price = Decimal(str(self.charts[asset].price_at(date)))
        asset_amount = amount / price

        if asset_amount > self.wallet[asset]:
            return False

        self.wallet[asset] -= asset_amount
        self.liquidity += self._add_percentage(amount, -self.sell_fee)

        return True

    def update(self, date: datetime):
        self._update_equities_history(date)

    def equities(self) -> tuple[list[datetime], list[Decimal]]:
        if len(self.equity_history["Date"].tolist()) == 0:
            date = list(self.charts.values())[0].timestamps()[0]
            return [date], [self._equity_at(date)]

        return (
            self.equity_history["Date"].tolist(),
            self.equity_history["Equity"].tolist(),
        )

    def budget_distribution(self, date: datetime) -> np.ndarray:
        total_equity = self._equity_at(date)

        distribution = np.zeros(len(self.charts.keys()), dtype=np.float32)
        for i, (asset, asset_amount) in enumerate(self.wallet.items()):
            price = Decimal(str(self.charts[asset].price_at(date)))
            asset_to_reference = asset_amount * price
            asset_allocation = asset_to_reference / total_equity
            distribution[i] = asset_allocation

        return distribution

    @staticmethod
    def _add_percentage(value: Decimal, percentage: Decimal) -> Decimal:
        """
        Add a percentage to a value.

        Args:
            value Decimal: The value to which the percentage is added.
            percentage Decimal: The percentage to be added.

        Returns:
            Decimal: The value with the added percentage.
        """
        add = percentage * Decimal(str(np.abs(value))) / Decimal("100.0")
        return value + add

    def _update_equities_history(self, date: datetime):
        current_equity = self._equity_at(date)
        if not (self.equity_history["Date"] == date).any():
            # Add a new row for the target date
            new_row = pd.DataFrame({"Date": [date], "Equity": [current_equity]})
            self.equity_history = pd.concat(
                [self.equity_history, new_row], ignore_index=True
            )

        else:
            self.equity_history.loc[
                self.equity_history["Date"] == date, "Equity"
            ] = current_equity

    def _equity_at(self, date: datetime, use_fee=False) -> Decimal:
        current_equity = self.liquidity

        for asset, asset_amount in self.wallet.items():
            price = Decimal(str(self.charts[asset].price_at(date)))
            current_equity += self._add_percentage(
                asset_amount * price, -self.sell_fee if use_fee else Decimal("0")
            )

        return current_equity
