"""
This module provides the implementation of a renderer.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from gym_trading.envs.chart import ChartDataFrame

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
    def render(
            self,
            chart: ChartDataFrame,
            buys: List[datetime],
            sells: List[datetime],
            now: datetime,
            equities: ChartDataFrame):
        """
        Render the chart with buy and sell signals and current equity information.

        Args:
            chart (ChartDataFrame): The chart data to render.
            buys (List[datetime]): The list of buy timestamps.
            sells (List[datetime]): The list of sell timestamps.
            now (datetime): The current timestamp.
            equities (ChartDataFrame): The equity data to display.

        Returns:
            None
        """


class PlotRenderer(Renderer):
    """Renders charts using matplotlib."""

    def render(
            self,
            chart: ChartDataFrame,
            buys: List[datetime],
            sells: List[datetime],
            now: datetime,
            equities: ChartDataFrame):
        """
        Render a chart with buy and sell markers, and equity information.

        Args:
            chart (ChartDataFrame): The price chart data.
            buys (List[datetime]): A list of buy timestamps.
            sells (List[datetime]): A list of sell timestamps.
            now (datetime): The current timestamp.
            equities (ChartDataFrame): The equity data.

        Returns:
            None
        """
        fig, axs = plt.subplots(2, 1, figsize=(15, 20))

        fig.suptitle(f'Total profit: {round(equities.profit(), 2)} %')

        axs[0].set_title('Prices')
        axs[0].plot(
            chart.dates,
            chart.values,
            alpha=0.7,
            label='Prices',
            zorder=1)
        axs[0].axvline(now, c='b', alpha=0.5, label='Today')

        axs[0].scatter(buys, [chart.value_at(date)
                       for date in buys], marker='^', c='g', label='BUY', zorder=2)
        axs[0].scatter(sells,
                       [chart.value_at(date) for date in sells],
                       marker='v',
                       c='r',
                       label='SELL',
                       zorder=2)
        axs[0].set_ylabel('Price')
        axs[0].set_xlabel('Time')
        axs[0].legend()

        axs[1].set_title('Equity')
        min_equity = min(equities.values) if len(
            equities.values) > 0 else 0
        axs[1].plot(equities.dates, equities.values,
                    alpha=0.7, label='Equities', zorder=1)
        axs[1].plot(chart.dates, [min_equity] * len(chart.dates),
                    alpha=0, linestyle='--', marker='None')
        axs[1].axvline(now, c='b', alpha=0.5, label='Today')
        for buy in buys:
            axs[1].axvline(buy, c='g', alpha=0.2, label='BUY')
        for sell in sells:
            axs[1].axvline(sell, c='r', alpha=0.2, label='SELL')

        axs[1].set_ylabel('Equity')
        axs[1].set_xlabel('Time')

        plt.show()
