import unittest
from datetime import datetime

import pandas as pd

from gym_trading.envs.chart import DataChart, AssetDataChart


class TestDataChart(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        data = {
            'Timestamp': ['2023-09-01', '2023-09-02', '2023-09-03'],
            'Value1': [10, 20, 30],
            'Value2': [100, 200, 300]
        }
        self.sample_dataset = pd.DataFrame(data)
        self.chart = DataChart(self.sample_dataset, 'Timestamp')

    def test_data_at(self):
        date = datetime(2023, 9, 2)
        result = self.chart.data_at(date)
        expected = pd.DataFrame({'Timestamp': datetime(2023, 9, 2), 'Value1': [20], 'Value2': [200]})
        pd.testing.assert_frame_equal(result, expected)

    def test_add(self):
        new_data = pd.DataFrame({'Timestamp': datetime(2023, 9, 4), 'Value1': [40], 'Value2': [400]})
        self.chart.add(new_data)
        expected_data = pd.concat([self.sample_dataset, new_data], ignore_index=True)
        pd.testing.assert_frame_equal(self.chart.dataset, expected_data)

    def test_window(self):
        start_date = datetime(2023, 9, 1)
        end_date = datetime(2023, 9, 3)
        result = self.chart.window(start_date, end_date)
        pd.testing.assert_frame_equal(result, self.sample_dataset)

    def test_timestamps(self):
        result = self.chart.timestamps()
        expected = [datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3)]
        self.assertEqual(result, expected)


class TestAssetDataChart(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        data = {
            'Timestamp': [datetime(2023, 9, 1), datetime(2023, 9, 2), datetime(2023, 9, 3)],
            'Price': [100, 200, 300]
        }
        self.sample_dataset = pd.DataFrame(data)
        self.asset_chart = AssetDataChart(self.sample_dataset, 'Timestamp', 'Price')

    def test_price_at(self):
        date = datetime(2023, 9, 2)
        result = self.asset_chart.price_at(date)
        expected = 200
        self.assertEqual(result, expected)

    def test_prices(self):
        result = self.asset_chart.prices()
        expected = pd.Series([100, 200, 300], name='Price')
        pd.testing.assert_series_equal(result, expected)
