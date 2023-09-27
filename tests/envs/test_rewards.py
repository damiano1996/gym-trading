import unittest
from datetime import datetime
from decimal import Decimal

import numpy as np

from gym_trading.envs.exchange import Exchange
from gym_trading.envs.rewards import ProfitRewarder, OneStepProfitRewarder


class TestProfitRewarder(unittest.TestCase):

    def test_reward(self):
        # Create a mock Exchange instance with equity data
        class MockExchange(Exchange):
            def market_buy(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def market_sell(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def update(self, date: datetime):
                pass

            def budget_distribution(self, date: datetime) -> np.ndarray:
                pass

            def reset(self):
                pass

            def equities(self):
                return ([datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3)],
                        [Decimal('1000'), Decimal('1100'), Decimal('1200')])

        exchange = MockExchange()
        rewarder = ProfitRewarder()
        result = rewarder.reward(exchange)
        expected = Decimal('20.0')
        self.assertEqual(result, expected)


class TestOneStepProfitRewarder(unittest.TestCase):

    def test_reward(self):
        # Create a mock Exchange instance with equity data
        class MockExchange(Exchange):
            def market_buy(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def market_sell(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def update(self, date: datetime):
                pass

            def budget_distribution(self, date: datetime) -> np.ndarray:
                pass

            def reset(self):
                pass

            def equities(self):
                return ([datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3)],
                        [Decimal('1000'), Decimal('1100'), Decimal('1200')])

        exchange = MockExchange()
        rewarder = OneStepProfitRewarder()
        result = rewarder.reward(exchange)
        expected = Decimal('9.090909091')
        self.assertAlmostEqual(result, expected)

    def test_reward_with_insufficient_data(self):
        # Create a mock Exchange instance with insufficient equity data
        class MockExchange(Exchange):
            def market_buy(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def market_sell(self, asset: str, amount: Decimal, date: datetime) -> bool:
                pass

            def update(self, date: datetime):
                pass

            def budget_distribution(self, date: datetime) -> np.ndarray:
                pass

            def reset(self):
                pass

            def equities(self):
                return ([datetime(2023, 9, 1)],
                        [Decimal('1000')])

        exchange = MockExchange()
        rewarder = OneStepProfitRewarder()
        result = rewarder.reward(exchange)
        expected = Decimal('0.0')
        self.assertEqual(result, expected)
