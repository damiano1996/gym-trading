"""
This module provides the implementation of a renderer.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Dict

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame
import seaborn as sns

from gym_trading.envs.chart import AssetDataChart
from gym_trading.envs.exchange import Exchange

# Constants for color palette
CB91_BLUE = "#2CBDFE"
CB91_GREEN = "#47DBCD"
CB91_PINK = "#F3A0F2"
CB91_PURPLE = "#9D2EC5"
CB91_VIOLET = "#661D98"
CB91_AMBER = "#F5B14C"

color_list = [CB91_BLUE, CB91_PINK, CB91_GREEN, CB91_AMBER, CB91_PURPLE, CB91_VIOLET]

# Configure seaborn for better visualization
sns.set(
    rc={
        "axes.prop_cycle": plt.cycler(color=color_list),
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "figure.figsize": (10, 5),
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "dimgrey",
        "xtick.bottom": False,
        "xtick.color": "dimgrey",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "dimgrey",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
    }
)

sns.set_context(
    "notebook", rc={"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18}
)


class Renderer(ABC):
    """Abstract base class for rendering implementations."""

    @abstractmethod
    def render_frame(
        self,
        charts: Dict[str, AssetDataChart],
        allocations_history: Tuple[List[datetime], List[np.ndarray]],
        now: datetime,
        exchange: Exchange,
    ):
        """
        Render the chart with buy and sell signals and current equity information.

        Args:
            charts (Dict[str, AssetDataChart]): A dictionary of asset data charts.
            allocations_history (Tuple[List[datetime], List[np.ndarray]]): A tuple
                containing history timestamps and allocations data.
            now (datetime): The current date and time.
            exchange (Exchange): The exchange object representing the trading environment.

        Returns:
            None
        """

    @abstractmethod
    def close(self):
        """Close the renderer."""


class MatPlotRenderer(Renderer):
    """Renders charts using matplotlib."""

    def render_frame(
        self,
        charts: Dict[str, AssetDataChart],
        allocations_history: Tuple[List[datetime], List[np.ndarray]],
        now: datetime,
        exchange: Exchange,
    ):
        """
        Render a frame with prices, budget allocation, and current equity information.

        Args:
            charts (Dict[str, AssetDataChart]): A dictionary of asset data charts.
            allocations_history (Tuple[List[datetime], List[np.ndarray]]): A tuple
                containing history timestamps and allocations data.
            now (datetime): The current date and time.
            exchange (Exchange): The exchange object representing the trading environment.

        Returns:
            None
        """
        make_figure(allocations_history, charts, exchange, now)
        plt.show()

    def close(self):
        """Close the MatPlotRenderer."""
        plt.close()


def make_figure(allocations_history, charts, exchange, now):
    """
    Create a Matplotlib figure with subplots for assets, budget allocation, and equity.

    Args:
        allocations_history (Tuple[List[datetime], List[np.ndarray]]): A tuple
            containing history timestamps and allocations data.
        charts (Dict[str, AssetDataChart]): A dictionary of asset data charts.
        exchange (Exchange): The exchange object representing the trading environment.
        now (datetime): The current date and time.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure.
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))

    equities = exchange.equities()

    profit = (equities[1][-1] / equities[1][0] - 1) * 100

    normalized_mrkt_prices = np.array(
        [chart.prices() / chart.prices().max() for chart in charts.values()]
    )
    average_normalized_mrkt_price = np.mean(normalized_mrkt_prices, axis=0)

    axs[0].plot(
        list(charts.values())[0].timestamps(),
        average_normalized_mrkt_price,
        alpha=1.0,
        label="Average Market",
        linestyle="--",
        zorder=1,
    )

    average_mrkt_profit = (
        average_normalized_mrkt_price[-1] / average_normalized_mrkt_price[0] - 1
    ) * 100.0

    fig.suptitle(
        f"Total Profit: {round(profit, 2)} %\n"
        f"vs\n"
        f"Average Market Profit: {round(average_mrkt_profit, 2)} %"
    )

    axs[0].set_title("Assets Prices")
    for i, (asset, chart) in enumerate(charts.items()):
        axs[0].plot(
            chart.timestamps(),
            chart.prices() / chart.prices().max(),
            alpha=0.4,
            label=asset,
            zorder=1,
        )
    axs[0].axvline(now, c="b", alpha=0.5, label="Today")
    axs[0].set_ylabel("Scaled Price")
    axs[0].set_xlabel("Time")
    axs[0].legend()

    axs[1].set_title("Budget Allocation")
    axs[1].stackplot(
        allocations_history[0],
        [
            [allocation[i] for allocation in allocations_history[1]]
            for i in range(len(charts.keys()))
        ],
        alpha=0.5,
        labels=list(charts.keys()),
    )
    axs[1].set_ylabel("Percentage")
    axs[1].set_xlabel("Time")
    axs[1].legend()

    axs[2].set_title("Portfolio Equity vs Average Market")
    axs[2].plot(
        equities[0],
        np.array(equities[1]) / equities[1][0],
        alpha=0.7,
        zorder=1,
        label="Portfolio Equity",
    )
    axs[2].plot(
        list(charts.values())[0].timestamps(),
        average_normalized_mrkt_price / average_normalized_mrkt_price[0],
        alpha=0.5,
        label="Average Market",
        linestyle="--",
        zorder=1,
    )
    axs[2].set_ylabel("Scaled Financial Amount")
    axs[2].set_xlabel("Time")
    axs[2].legend()
    return fig


class PyGamePlotRenderer(Renderer):
    """Renders charts using pygame and matplotlib."""

    def __init__(self, render_fps=4):
        """
        Initialize the PyGamePlotRenderer.

        Args:
            render_fps (int): The desired frame rate for rendering.
        """
        super(PyGamePlotRenderer, self).__init__()
        self.screen = None
        self.clock = None
        self.render_fps = render_fps

        self.matplot_renderer = MatPlotRenderer()

    def render_frame(
        self,
        charts: Dict[str, AssetDataChart],
        allocations_history: Tuple[List[datetime], List[np.ndarray]],
        now: datetime,
        exchange: Exchange,
    ):
        """
        Render a chart with prices, budget allocation, and current equity information.

        Args:
            charts (Dict[str, AssetDataChart]): A dictionary of asset data charts.
            allocations_history (Tuple[List[datetime], List[np.ndarray]]): A tuple
                containing history timestamps and allocations data.
            now (datetime): The current date and time.
            exchange (Exchange): The exchange object representing the trading environment.

        Returns:
            None
        """
        fig = make_figure(allocations_history, charts, exchange, now)

        # plt.show()
        plt.close()

        # Render the figure as an image
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()

        self._init_screen(fig)

        self._init_clock()

        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(raw_data, size, "RGBA")

        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        self.clock.tick(self.render_fps)

    def _init_clock(self):
        """Initialize the Pygame clock."""
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _init_screen(self, fig):
        """Initialize the Pygame screen."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            width, height = fig.get_size_inches() * fig.get_dpi()
            pygame.display.set_mode((int(width), int(height)))
            pygame.display.set_caption("Trading Gym")
            self.screen = pygame.display.get_surface()

    def close(self):
        """Close the PyGamePlotRenderer."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
