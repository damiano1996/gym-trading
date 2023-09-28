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

CB91_BLUE = '#2CBDFE'
CB91_GREEN = '#47DBCD'
CB91_PINK = '#F3A0F2'
CB91_PURPLE = '#9D2EC5'
CB91_VIOLET = '#661D98'
CB91_AMBER = '#F5B14C'

color_list = [CB91_BLUE, CB91_PINK, CB91_GREEN, CB91_AMBER,
              CB91_PURPLE, CB91_VIOLET]

sns.set(rc={'axes.prop_cycle': plt.cycler(color=color_list),
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'figure.figsize': (10, 5),
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False})

sns.set_context("notebook", rc={"font.size": 16,
                                "axes.titlesize": 20, "axes.labelsize": 18})


class Renderer(ABC):
    """Abstract base class for rendering implementations."""

    @abstractmethod
    def render_frame(
            self,
            charts: Dict[str, AssetDataChart],
            allocations_history: Tuple[List[datetime], List[np.ndarray]],
            now: datetime,
            exchange: Exchange
    ):
        """
        Render the chart with buy and sell signals and current equity information.
        """

    @abstractmethod
    def close(self):
        """ To close the renderer """


class MatPlotRenderer(Renderer):
    """Renders charts using matplotlib."""

    def render_frame(
            self,
            charts: Dict[str, AssetDataChart],
            allocations_history: Tuple[List[datetime], List[np.ndarray]],
            now: datetime,
            exchange: Exchange
    ):
        make_figure(allocations_history, charts, exchange, now)
        plt.show()

    def close(self):
        plt.close()


def make_figure(allocations_history, charts, exchange, now):
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))

    equities = exchange.equities()

    profit = (equities[1][-1] / equities[1][0] - 1) * 100
    fig.suptitle(f'Total profit: {round(profit, 2)} %')

    axs[0].set_title('Assets')
    for i, (asset, chart) in enumerate(charts.items()):
        axs[0].plot(
            chart.timestamps(),
            chart.prices() / chart.prices().max(),
            alpha=0.7,
            label=asset,
            zorder=1)
    axs[0].axvline(now, c='b', alpha=0.5, label='Today')
    axs[0].set_ylabel('Normalized Price')
    axs[0].set_xlabel('Time')
    axs[0].legend()

    axs[1].set_title('Budget Allocation')
    axs[1].stackplot(
        allocations_history[0],
        [[allocation[i] for allocation in allocations_history[1]] for i in range(len(charts.keys()))],
        alpha=0.5,
        labels=list(charts.keys()))
    axs[1].set_ylabel('Allocation Percentage')
    axs[1].set_xlabel('Time')
    axs[1].legend()

    axs[2].set_title('Equity')
    axs[2].plot(equities[0], equities[1], alpha=0.7, zorder=1)
    axs[2].set_ylabel('Equity')
    axs[2].set_xlabel('Time')
    return fig


class PyGamePlotRenderer(Renderer):
    """Renders charts using matplotlib."""

    def __init__(self, render_fps=4):
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
            exchange: Exchange
    ):
        """
        Render a chart with buy and sell markers, and equity information.

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
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _init_screen(self, fig):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            width, height = fig.get_size_inches() * fig.get_dpi()
            pygame.display.set_mode((int(width), int(height)))
            pygame.display.set_caption("Trading Gym")
            self.screen = pygame.display.get_surface()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
