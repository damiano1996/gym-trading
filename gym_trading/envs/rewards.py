from abc import ABC, abstractmethod

from gym_trading.envs.exchange import Exchange


class Rewarder(ABC):
    """
    Abstract base class for reward calculation.
    """

    @abstractmethod
    def reward(self, exchange: Exchange) -> float:
        """
        Calculate the reward based on the provided exchange data.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            float: The calculated reward.
        """
        pass


class ProfitRewarder(Rewarder):
    """
    Calculates the reward based on profit relative to the initial equity.
    """

    def reward(self, exchange: Exchange) -> float:
        """
        Calculate the profit-based reward.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            float: The calculated reward.
        """
        equities = exchange.equities()[1]
        return (equities[-1] / equities[0] - 1) * 100


class OneStepProfitRewarder(Rewarder):
    """
    Calculates the reward based on profit relative to the previous equity step.
    """

    def reward(self, exchange: Exchange) -> float:
        """
        Calculate the one-step profit-based reward.

        Args:
            exchange (Exchange): The exchange data.

        Returns:
            float: The calculated reward.
        """
        equities = exchange.equities()[1]
        if len(equities) < 2:
            return 0.0
        return (equities[-1] / equities[-2] - 1) * 100
