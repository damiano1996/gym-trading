"""
This module provides the implementation of a trading environment.
"""

from datetime import datetime
from typing import List, Any, Tuple

import gym
from gym import Space
from gym.core import ActType
from gym.spaces import Discrete

from gym_trading.envs.data_loader import DataLoader
from gym_trading.envs.exchange import Exchange
from gym_trading.envs.renderer import Renderer
from gym_trading.envs.rewards import Rewarder


class TradingEnv(gym.Env):
    """Custom Gym environment for trading."""

    ACTIONS = ["BUY", "SELL", "HOLD"]

    def __init__(
            self,
            data_loader: DataLoader,
            exchange: Exchange,
            rewarder: Rewarder,
            renderer: Renderer,
            observation_window_size=10
    ):
        """
        Initialize the TradingEnv.

        Args:
            data_loader (DataLoader): The data loader to load the trading data.
            exchange (Exchange): The exchange object for trading actions.
            rewarder (Rewarder): The rewarder object to compute rewards.
            renderer (Renderer): The renderer object to visualize the trading environment.
            observation_window_size (int, optional): The size of the observation window. Defaults to 10.
        """
        self.data = data_loader.load()
        self.exchange = exchange
        self.rewarder = rewarder
        self.renderer = renderer

        self.observation_window_size = observation_window_size

        self.buys: List[datetime] = []
        self.sells: List[datetime] = []

        self.date_index = observation_window_size - 1

        self.last_observation = self.reset()[0]
        self.observation_space = Space(
            shape=self.last_observation.shape,
            dtype=self.last_observation.dtypes)
        self.action_space = Discrete(len(self.ACTIONS))

    def step(self, action: ActType) -> Tuple[Any, float, bool, bool, dict]:
        """
        Perform a step in the trading environment.

        Args:
            action (ActType): The action to take in the environment.

        Returns:
            Tuple: A tuple containing the next observation, reward, done flag, info, and action space.
        """

        if action in ('BUY', 0):
            if self.exchange.buy_all(
                    self.data.value_at(
                        self._now()),
                    self._now()):
                self.buys.append(self._now())
        elif action in ('SELL', 1):
            if self.exchange.sell_all(
                    self.data.value_at(
                        self._now()),
                    self._now()):
                self.sells.append(self._now())
        else:
            self.exchange.hold(self.data.value_at(self._now()), self._now())

        self.date_index += 1
        done = False
        if self.date_index == len(self.data.dates):
            # Force sell before exit
            if self.exchange.sell_all(
                    self.data.value_at(
                        self._now()),
                    self._now()):
                self.sells.append(self._now())
            done = True

        observation = self.data.with_data_window_until_date(
            self._now(), self.observation_window_size)

        reward = self.rewarder.reward(self.exchange.get_equity_history())

        return observation, reward, done, False, {}

    def reset(self, **_kwargs):
        """
        Reset the trading environment.

        Returns:
            Tuple: A tuple containing the initial observation and info.
        """
        self.buys: List[datetime] = []
        self.sells: List[datetime] = []

        self.date_index = self.observation_window_size - 1
        self.exchange.reset()

        return self.data.with_data_window_until_date(
            self._now(), self.observation_window_size), {}

    def _now(self) -> datetime:
        """
        Get the current timestamp.

        Returns:
            datetime: The current timestamp.
        """
        return self.data.dates[min(self.date_index, len(self.data.dates) - 1)]

    def render(self):
        """Render the trading environment."""
        self.renderer.render(self.data, self.buys, self.sells,
                             self._now(), self.exchange.get_equity_history())
