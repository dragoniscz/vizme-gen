import numpy as np
import pandas as pd

from . import VisualizationPipeline

INVERT_PARAM = 'invert'


class TableVisualizationPipeline(VisualizationPipeline):

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    def transform_one(self, data: pd.Series, output: str) -> None:
        with open(output.replace('.png', '.html'), 'w') as file:
            frame = data.to_frame()
            if INVERT_PARAM in self._parameters and self._parameters[INVERT_PARAM]:
                frame = frame.transpose()
            frame.to_html(file, index=False, float_format='%.2f')