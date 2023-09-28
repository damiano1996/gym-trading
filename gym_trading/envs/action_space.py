from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy._typing import NDArray


class BudgetAllocationSpace(Box):
    """
    Custom Gym space for budget allocation.

    This class defines a custom Gym space for representing budget allocation. It inherits from the Box space and enforces
    that the allocation vector is within the range [0, 1] and sums up to 1.

    Parameters:
        num_assets (int): The number of assets in the allocation.

    Example usage:
        space = BudgetAllocationSpace(num_assets=3)
        action = space.sample()
    """

    def __init__(self, num_assets):
        """
        Initialize the BudgetAllocationSpace.

        Args:
            num_assets (int): The number of assets in the allocation.
        """
        super().__init__(
            low=np.zeros(num_assets, dtype=np.float32),
            high=np.ones(num_assets, dtype=np.float32),
            shape=(num_assets,),
        )

    def sample(self, mask: None = None) -> NDArray[Any]:
        """
        Generate a normalized random sample within the defined space.

        This method generates a random sample within the defined space, typically used for generating initial action
        values in reinforcement learning tasks. The generated sample is then normalized so that the sum of its components
        equals 1.

        Args:
            mask: An optional mask that can be applied to restrict the sampling.

        Returns:
            NDArray[Any]: A normalized random sample within the space.
        """
        sample = super().sample(mask)
        normalized_sample = sample / np.sum(sample)
        return normalized_sample
