import unittest

import numpy as np

from gym_trading.envs.action_space import BudgetAllocationSpace


class TestBudgetAllocationSpace(unittest.TestCase):

    def test_initialization(self):
        num_assets = 3
        space = BudgetAllocationSpace(num_assets)

        # Check if the low and high values are correct
        self.assertTrue(np.all(space.low == np.zeros(num_assets, dtype=np.float32)))
        self.assertTrue(np.all(space.high == np.ones(num_assets, dtype=np.float32)))

        # Check if the shape is set correctly
        self.assertEqual(space.shape, (num_assets,))

    def test_sample(self):
        num_assets = 4
        space = BudgetAllocationSpace(num_assets)

        # Create a mask with all assets included
        mask = None

        # Sample from the space
        sample = space.sample(mask)

        # Check if the sample is a valid probability distribution
        self.assertTrue(np.allclose(sample, sample / np.sum(sample)))
