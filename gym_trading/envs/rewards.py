"""
This module provides the implementation of a rewarder.
"""

from abc import ABC, abstractmethod

from gym_trading.envs.chart import ChartDataFrame


class Rewarder(ABC):
    """Abstract base class for reward computation implementations."""

    @abstractmethod
    def reward(self, equities: ChartDataFrame) -> float:
        """
        Compute the reward based on the given equity data.

        Args:
            equities (ChartDataFrame): The equity data to compute the reward from.

        Returns:
            float: The computed reward.
        """


class ProfitRewarder(Rewarder):
    """Rewarder implementation that computes the profit as the reward."""

    def reward(self, equities: ChartDataFrame) -> float:
        """
        Compute the reward as the total profit from the given equity data.

        Args:
            equities (ChartDataFrame): The equity data to compute the reward from.

        Returns:
            float: The computed profit as the reward.
        """
        return equities.profit()
