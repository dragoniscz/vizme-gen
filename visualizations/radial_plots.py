import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from loguru import logger

from util.dataset import get_feature_ordering_order_based_on_correlation
from . import VisualizationPipeline

# parameters
COLOR_MAP_PATH_PARAM = 'colorMapPath'
COLOR_MAP_PARAM = 'colorMap'
SCALE_COEFFICIENT_PARAM = 'scaleCoeff'
BOTTOM_MARGIN_PARAM = 'bottomMargin'
CORRELATION_ORDERING = 'correlationOrdering'

# custom colorm maps
top = cm.get_cmap('Blues_r', 128)
bottom = cm.get_cmap('Oranges', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
orangeBlueCmap = ListedColormap(newcolors, name='OrangeBlue')

customColorMaps = {'OrangeBlue': orangeBlueCmap}


def _rescale(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))


class RadialPlotsVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()
        self._parameters = None
        self._colorMap = None
        self._M = None
        self._width = None
        self._ordering = None
        self._angles = None
        self._rescale = None
        self._scaleCoefficient = 100
        self._bottomMargin = 0

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        self._M = data.shape[1]
        self._width = 2 * np.pi / self._M
        self._indexes = list(range(1, self._M + 1))
        self._angles = [i * self._width for i in self._indexes]
        self._rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        if COLOR_MAP_PATH_PARAM in self._parameters:
            colorMapPath = self._parameters[COLOR_MAP_PATH_PARAM]
            self._colorMap = pd.read_csv(colorMapPath, header=None).sort_values(by=1)
            self._ordering = self._colorMap.index
            self._colorMap = self._colorMap[1]

        # TODO ring mode
        if CORRELATION_ORDERING in self._parameters and self._parameters[CORRELATION_ORDERING]:
            try:
                self._ordering = get_feature_ordering_order_based_on_correlation(data)
            except ValueError:
                logger.warning("Could not compute ordering based on correlation.")

        if COLOR_MAP_PARAM in self._parameters:
            color_map = self._parameters[COLOR_MAP_PARAM]
            if color_map in customColorMaps:
                self._colorMap = customColorMaps[color_map]
            else:
                self._colorMap = color_map

        if SCALE_COEFFICIENT_PARAM in self._parameters:
            self._scaleCoefficient = self._parameters[SCALE_COEFFICIENT_PARAM]

        if BOTTOM_MARGIN_PARAM in self._parameters:
            self._bottomMargin = self._parameters[BOTTOM_MARGIN_PARAM]

    def transform_one(self, data: pd.Series, output: str) -> None:

        fig = plt.figure(figsize=(4, 4))
        ax = plt.subplot(111, projection='polar')

        data_h = (data * self._scaleCoefficient).to_numpy()

        if isinstance(self._colorMap, ListedColormap):
            color = self._colorMap(_rescale(data_h))
        else:
            color = self._colorMap

        ax.bar(x=self._angles, height=data_h, width=self._width, bottom=self._bottomMargin, linewidth=0,
               edgecolor="white", color=color)
        # ax.plot(angles + [angles[0]], sample_h.to_numpy(),)
        ax.set_rorigin(-1)

        ax.axis('off')
        plt.savefig(output, pad_inches=0, bbox_inches='tight', transparent=False)
        plt.close(fig)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        # if self._coords is None:
        #     raise SyntaxError('Method ::fit must called before ::transform.')
        if self._ordering is not None:
            data = data.iloc[:, self._ordering]
        super().transform(data, labels, output)


