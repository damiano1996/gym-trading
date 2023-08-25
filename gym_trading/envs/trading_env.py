"""
This module provides the implementation of a trading environment.
"""

from datetime import datetime
from typing import List, Any, Tuple

import gym
import numpy as np
import pandas as pd
from gym.core import ActType
from gym.spaces import Box
from pandas import DataFrame

from gym_trading.envs.action_space import BudgetAllocationSpace
from gym_trading.envs.data_loader import AssetChartDataLoader
from gym_trading.envs.exchange import Exchange
from gym_trading.envs.renderer import Renderer
from gym_trading.envs.rewards import Rewarder


class TradingEnv(gym.Env):
    """Custom Gym environment for trading."""

    def __init__(
            self,
            data_loader: AssetChartDataLoader,
            exchange: Exchange,
            rewarder: Rewarder,
            renderer: Renderer,
    ):
        """
        Initialize the TradingEnv.

        Args:
            data_loader (AssetChartDataLoader): The data loader to load the trading data.
            exchange (Exchange): The exchange object for trading actions.
            rewarder (Rewarder): The rewarder object to compute rewards.
            renderer (Renderer): The renderer object to visualize the trading environment.
        """
        self.charts = data_loader.load()
        self._validate_charts()

        self.exchange = exchange
        self.rewarder = rewarder
        self.renderer = renderer

        self.allocations_history: Tuple[List[datetime], List[np.ndarray]] = ([], [])

        self.obs_index = 0

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.reset()[0].shape)
        self.action_space = BudgetAllocationSpace(len(self.charts.keys()))

    def step(self, action: ActType) -> tuple[DataFrame | Any, float, bool, dict[Any, Any]]:
        assert action.shape == self.action_space.shape, f'Expected action shape: {self.action_space.shape}, but {action.shape} was given.'

        target_allocation = action

        current_allocation = self.exchange.budget_distribution(self._now())

        self.allocations_history[0].append(self._now())
        self.allocations_history[1].append(current_allocation)

        current_equity = self.exchange.equities()[1][-1]

        diff_allocation = target_allocation - current_allocation

        for i, asset_diff in enumerate(diff_allocation):
            asset_name = list(self.charts.keys())[i]

            if asset_diff > 0:
                self.exchange.market_buy(asset_name, np.abs(asset_diff * current_equity), self._now())
            elif asset_diff < 0:
                self.exchange.market_sell(asset_name, np.abs(asset_diff * current_equity), self._now())

        self.obs_index += 1
        done = False
        if self.obs_index >= len(self._get_timestamps()):
            done = True

        observation = self._get_next_obs()

        reward = self.rewarder.reward(self.exchange)

        return observation, reward, done, {}

    def reset(self, **_kwargs):
        self.allocations_history: Tuple[List[datetime], List[np.ndarray]] = ([], [])

        self.obs_index = 0
        self.exchange.reset()

        obs = self._get_next_obs()
        return obs

    def _get_next_obs(self):
        result = []
        for name, chart in self.charts.items():
            data = chart.data_at(self._now())
            if data is not None and not data.empty:
                data = data.drop(columns=[chart.timestamp_column_name])
                result.append(data)
            else:
                raise IndexError(f'Unable to create an observation for date: {self._now()}')

        return pd.concat(result, ignore_index=True, axis=1).to_numpy(dtype=np.float32)

    def _get_timestamps(self):
        timestamps = list(self.charts.values())[0].timestamps()
        return timestamps

    def _now(self) -> datetime:
        timestamps = self._get_timestamps()
        return timestamps[min(self.obs_index, len(timestamps) - 1)]

    def render(self, mode: str = 'human'):
        """Render the trading environment."""
        return self.renderer.render(self.charts, self.allocations_history, self._now(), self.exchange)

    def _validate_charts(self):
        if len(self.charts) == 0:
            raise AssertionError('At least one dataframe must be given.')

        first_chart = list(self.charts.values())[0]
        unique_dates = set(first_chart.timestamps())

        for chart in list(self.charts.values())[1:]:
            if set(chart.timestamps()) != unique_dates:
                raise AssertionError('All datasets must have same timestamps.')
