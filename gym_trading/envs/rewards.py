from abc import ABC, abstractmethod
from decimal import Decimal

from gym_trading.envs.exchange import Exchange


class Rewarder(ABC):
    """
    Abstract base class for reward calculation.
    """

    @abstractmethod
    def reward(self, exchange: Exchange) -> Decimal:
        """
        Calculate the reward based on the provided exchange data.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            Decimal: The calculated reward.
        """


class ProfitRewarder(Rewarder):
    """
    Calculates the reward based on profit relative to the initial equity.
    """

    def reward(self, exchange: Exchange) -> Decimal:
        """
        Calculate the profit-based reward.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            Decimal: The calculated reward.
        """
        equities = exchange.equities()[1]
        return (Decimal(equities[-1]) / Decimal(equities[0]) - Decimal("1")) * Decimal(
            "100"
        )


class OneStepProfitRewarder(Rewarder):
    """
    Calculates the reward based on profit relative to the previous equity step.
    """

    def reward(self, exchange: Exchange) -> Decimal:
        """
        Calculate the one-step profit-based reward.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            Decimal: The calculated reward.
        """
        equities = exchange.equities()[1]
        if len(equities) < 2:
            return Decimal("0.0")
        return (Decimal(equities[-1]) / Decimal(equities[-2]) - Decimal("1")) * Decimal(
            "100"
        )
