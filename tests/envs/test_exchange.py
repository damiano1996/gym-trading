import unittest
from datetime import datetime
from decimal import Decimal

import numpy as np

from gym_trading.envs.data_loader import ListAssetChartDataLoader
from gym_trading.envs.exchange import BaseExchange


class TestDataChart(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset using ListAssetChartDataLoader
        dates = [datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3)]
        prices = [100.0, 200.0, 300.0]
        asset_name = 'Asset1'
        data_loader = ListAssetChartDataLoader(dates, prices, asset_name)
        self.exchange = BaseExchange(data_loader, init_liquidity=Decimal('1000.0'), buy_fee=Decimal('0.1'),
                                     sell_fee=Decimal('0.1'))

    def test_market_buy(self):
        date = datetime(2023, 9, 1)
        asset = 'Asset1'
        amount = Decimal(100.0)
        result = self.exchange.market_buy(asset, amount, date)
        self.assertTrue(result)
        self.assertEqual(self.exchange.liquidity, Decimal('900.0'))
        self.assertEqual(self.exchange.wallet[asset], Decimal('0.999'))  # (amount / price) - 0.1%

    def test_reset(self):
        self.exchange.reset()
        self.assertEqual(self.exchange.liquidity, Decimal('1000.0'))
        self.assertEqual(len(self.exchange.wallet), 1)
        self.assertEqual(len(self.exchange.equity_history), 0)

    def test_market_sell(self):
        date = datetime(2023, 9, 1)
        asset = 'Asset1'
        self.exchange.market_buy(asset, Decimal('60.0'), date)

        amount = Decimal('50.0')
        result = self.exchange.market_sell(asset, amount, date)
        self.assertTrue(result)
        self.assertEqual(self.exchange.liquidity, Decimal('989.95'))  # liquidity - 60 + (50 - 0.1%)
        self.assertEqual(
            self.exchange.wallet[asset],
            Decimal('0.0994'))  # (buy_amount / price) -0.1% - sell_amount/price

    def test_update(self):
        date = datetime(2023, 9, 1)
        self.exchange.update(date)
        _, equities = self.exchange.equities()
        self.assertEqual(equities[0], Decimal('1000.0'))

    def test_budget_distribution(self):
        date = datetime(2023, 9, 1)
        distribution = self.exchange.budget_distribution(date)
        self.assertTrue(np.allclose(distribution, np.array([0.0])))

    def test_add_percentage(self):
        value = Decimal('100.0')
        percentage = Decimal('10.0')
        result = self.exchange._add_percentage(value, percentage)
        self.assertEqual(result, Decimal('110.0'))
