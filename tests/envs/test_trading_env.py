import unittest
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, List

import numpy as np

from gym_trading.envs import TradingEnv
from gym_trading.envs.chart import AssetDataChart
from gym_trading.envs.data_loader import ListAssetChartDataLoader
from gym_trading.envs.exchange import BaseExchange, Exchange
from gym_trading.envs.renderer import Renderer
from gym_trading.envs.rewards import ProfitRewarder


class MockRenderer(Renderer):

    def render_frame(
            self,
            charts: Dict[str, AssetDataChart],
            allocations_history: Tuple[List[datetime], List[np.ndarray]],
            now: datetime,
            exchange: Exchange
    ):
        pass

    def close(self):
        pass


class TestTradingEnv(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset using ListAssetChartDataLoader
        dates = [datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3), datetime(2023, 9, 4)]
        prices = [100.0, 200.0, 300.0, 150.0]
        asset_name = 'Asset1'
        data_loader = ListAssetChartDataLoader(dates, prices, asset_name)

        exchange = BaseExchange(
            data_loader,
            init_liquidity=Decimal('1000.0'),
            buy_fee=Decimal('0.1'),
            sell_fee=Decimal('0.1')
        )

        rewarder = ProfitRewarder()
        renderer = MockRenderer()

        self.env = TradingEnv(data_loader, exchange, rewarder, renderer=renderer)

    def test_reset__returns_expected_types(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertEqual(info, {})

    def test_reset__should_return_the_first_price(self):
        obs, info = self.env.reset()
        self.assertEqual([[100.0]], obs)

    def test_step__returns_expected_types(self):
        obs, info = self.env.reset()
        action = self.env.action_space.sample()  # Sample a random action

        # Take a step
        next_obs, reward, done, _, _ = self.env.step(action)

        self.assertEqual(next_obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, Decimal)
        self.assertIsInstance(done, bool)
        self.assertEqual(info, {})

    def test_step__should_return_prices(self):
        obs, info = self.env.reset()

        self.assertEqual([[100.0]], obs)
        # price:100 -> invest:0 --> next_price:200 -> reward:0.0
        self._assert_step_outputs(np.array([0]), [[200.0]], Decimal('0'), False)
        # price:200 -> invest:0.5 --> next_price:300 -> reward:24.925
        self._assert_step_outputs(np.array([0.5]), [[300.0]], Decimal('24.925'), False)
        # price:300 -> invest:1.0 --> next_price:150 -> reward:−37.562
        self._assert_step_outputs(np.array([1.0]), [[150.0]], Decimal('-37.562'), False)
        # price:150 -> invest:0.0 --> next_price: 150 -> force sell since done -> reward:-37.625
        self._assert_step_outputs(np.array([0.0]), [[150.0]], Decimal('-37.625'), True)

    def _assert_step_outputs(self, action, expected_obs, expected_reward, expected_done):
        actual_obs, actual_reward, actual_done, _, _ = self.env.step(action)
        self.assertEqual(expected_obs, actual_obs)
        self.assertAlmostEqual(expected_reward, actual_reward, places=3)
        self.assertEqual(expected_done, actual_done)
