import json
import numpy as np
import pandas as pd

from . import VisualizationPipeline


class ParallelCoordinatesVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()
        self._color = [1, 1, 1]
        self._opacity = 0.5
        self._stroke_width = 2
        self._bounds = {}

    def setup(self, parameters = None) -> None:
        super().setup(parameters)
        if self._parameters['color']:
            self._color = self._parameters['color']
        if self._parameters['opacity']:
            self._opacity = float(self._parameters['opacity'])


    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        for column in data.columns:
            self._bounds[column] = (data[column].min(), data[column].max())

    def transform_one(self, data: pd.DataFrame, output: str) -> None:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Parcoords(
            dimensions=list(map(
                lambda column: dict(
                    range=self._bounds[column],
                    label=None,
                    values=(data[column],),
                ),
                self._bounds
            )),
            line=dict(
                color=f"rgba({self._color[0]}, {self._color[1]}, {self._color[2]}, {self._color[3] if len(self._color) > 3 else self._opacity})",
            ),
        ))

        fig.write_image(output)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        if self._bounds is None:
            raise SyntaxError('Method ::fit must called before ::transform.')
        super().transform(data, labels, output)



