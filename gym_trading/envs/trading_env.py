"""
This module provides the implementation of a trading environment.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Any, Tuple, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.core import ActType
from gymnasium.experimental.functional import ObsType
from gymnasium.spaces import Box

from gym_trading.envs.action_space import BudgetAllocationSpace
from gym_trading.envs.data_loader import AssetChartDataLoader
from gym_trading.envs.exchange import Exchange
from gym_trading.envs.renderer import Renderer, MatPlotRenderer
from gym_trading.envs.rewards import Rewarder


class TradingEnv(gym.Env):
    """Custom Gym environment for trading."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            data_loader: AssetChartDataLoader,
            exchange: Exchange,
            rewarder: Rewarder,
            renderer: Renderer = None,
            final_report_plot: bool = False,
    ):
        """
        Initialize the TradingEnv.

        Args:
            data_loader (AssetChartDataLoader): The data loader to load the trading data.
            exchange (Exchange): The exchange object for trading actions.
            rewarder (Rewarder): The rewarder object to compute rewards.
            renderer (Renderer): The renderer object to visualize the trading environment.
            final_report_plot (bool): To create a final report.
        """
        self.charts = data_loader.load()
        self._validate_charts()

        self.exchange = exchange
        self.rewarder = rewarder
        self.renderer = renderer
        self.final_report_plot = final_report_plot

        self.allocations_history: Tuple[List[datetime], List[np.ndarray]] = ([], [])

        self.obs_index = 0

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.reset()[0].shape)
        self.action_space = BudgetAllocationSpace(len(self.charts.keys()))

        if self.renderer is not None:
            self.metadata['render_modes'].append(self.renderer.__class__.__name__)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert action.shape == self.action_space.shape, f'Expected action shape: {self.action_space.shape}, but {action.shape} was given.'

        truncated = False
        info = {}

        target_allocation = action

        current_allocation = self.exchange.budget_distribution(self._now())

        current_equity = self.exchange.equities()[1][-1]

        diff_allocation = target_allocation - current_allocation

        for i, asset_diff in enumerate(diff_allocation):
            asset_name = list(self.charts.keys())[i]

            asset_diff = Decimal(str(asset_diff))

            if asset_diff > Decimal('0'):
                self.exchange.market_buy(asset_name, abs(asset_diff * current_equity), self._now())
            elif asset_diff < Decimal('0'):
                self.exchange.market_sell(asset_name, abs(asset_diff * current_equity), self._now())

        # ---------- NEXT DAY ----------

        self.obs_index += 1
        done = False
        if self.obs_index >= len(self._get_timestamps()):
            done = True

        self.exchange.update(self._now())

        actual_allocation = self.exchange.budget_distribution(self._now())
        self.allocations_history[0].append(self._now())
        self.allocations_history[1].append(actual_allocation)

        observation = self._get_obs()

        reward = self.rewarder.reward(self.exchange)

        self._render_frame()

        if done:
            self._make_final_report()

        return observation, reward, done, truncated, info

    def _make_final_report(self):
        if self.final_report_plot:
            matplot_renderer = MatPlotRenderer()
            matplot_renderer.render_frame(self.charts, self.allocations_history, self._now(), self.exchange)
            matplot_renderer.close()

    def _render_frame(self):
        if self.renderer is not None:
            self.renderer.render_frame(self.charts, self.allocations_history, self._now(), self.exchange)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        self.allocations_history: Tuple[List[datetime], List[np.ndarray]] = ([], [])

        self.obs_index = 0
        self.exchange.reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    @staticmethod
    def _get_info():
        return {}

    def _get_obs(self):
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
        self._render_frame()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def _validate_charts(self):
        if len(self.charts) == 0:
            raise AssertionError('At least one dataframe must be given.')

        first_chart = list(self.charts.values())[0]
        unique_dates = set(first_chart.timestamps())

        for chart in list(self.charts.values())[1:]:
            if set(chart.timestamps()) != unique_dates:
                raise AssertionError('All datasets must have same timestamps.')
