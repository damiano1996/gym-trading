"""
This module provides the implementation of a rewarder.
"""

from abc import ABC, abstractmethod

from gym_trading.envs.exchange import Exchange


class Rewarder(ABC):
    """Abstract base class for reward computation implementations."""

    @abstractmethod
    def reward(self, exchange: Exchange) -> float:
        """
        Computes the reward based on the status of the exchange.
        """


class ProfitRewarder(Rewarder):
    """Rewarder implementation that computes the profit as the reward."""

    def reward(self, exchange: Exchange) -> float:
        equities = exchange.equities()[1]
        return (equities[-1] / equities[0] - 1) * 100
