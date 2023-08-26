import numpy as np
from gymnasium.spaces import Box


class BudgetAllocationSpace(Box):

    def __init__(self, num_assets):
        super().__init__(
            low=np.zeros(num_assets, dtype=np.float32),
            high=np.ones(num_assets, dtype=np.float32),
            shape=(num_assets,))
