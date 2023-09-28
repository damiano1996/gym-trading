from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy._typing import NDArray


class BudgetAllocationSpace(Box):
    def __init__(self, num_assets):
        super().__init__(
            low=np.zeros(num_assets, dtype=np.float32),
            high=np.ones(num_assets, dtype=np.float32),
            shape=(num_assets,),
        )

    def sample(self, mask: None = None) -> NDArray[Any]:
        sample = super().sample(mask)
        normalized_sample = sample / np.sum(sample)
        return normalized_sample
