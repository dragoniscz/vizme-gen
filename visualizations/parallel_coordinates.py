import json
import numpy as np
import pandas as pd

from . import VisualizationPipeline


class ParallelCoordinatesVisualizationPipeline(VisualizationPipeline):
    def __init__(self, opacity = 0.1, stroke_width = 2, skip_existing: bool = False):
        super().__init__(skip_existing)
        self._opacity = opacity
        self._stroke_width = stroke_width
        self._bounds = {}

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        for column in data.columns:
            self._bounds[column] = (data[column].min(), data[column].max())

    def transform_one(self, data: pd.DataFrame, output: str) -> None:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Parcoords(
            line_color='black',
            dimensions=list(map(
                lambda column: dict(
                    range=self._bounds[column],
                    label=None,
                    values=(data[column],),
                ),
                self._bounds
            ))
        ))

        fig.write_image(output)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        if self._bounds is None:
            raise SyntaxError('Method ::fit must called before ::transform.')
        super().transform(data, labels, output)



