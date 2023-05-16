import json
import numpy as np
import pandas as pd
from loguru import logger

from . import VisualizationPipeline
from . import create_folder, file_exists


class ParallelCoordinatesVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()
        self._color = [1, 1, 1, 0]
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

    def transform_one(self, data: pd.Series, output: str) -> None:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Parcoords(
            dimensions=list(map(
                lambda column: dict(
                    range=self._bounds[column],
                    label=None,
                    values=data[column] if len(data.shape) == 2 else (data[column],),
                ),
                self._bounds
            )),
            line=dict(
                color=f"rgba({self._color[0]}, {self._color[1]}, {self._color[2]}, {self._color[3] if len(self._color) > 3 else self._opacity})",
            ),
        ))

        fig.write_image(output)

    def transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        if grouping != 'special':
            super().transform_group(data, labels, output, grouping)
            return

        logger.info("Creating output dir: {0}", output)
        create_folder(output)

        labels_counts = labels.value_counts().apply(lambda x: x if x <= self._n_samples else self._n_samples)
        total_samples = labels_counts.sum()
        logger.info("Found {0} target classes, {1} samples at total.", labels_counts.index.shape[0], total_samples)

        groups = {}
        for target_class in labels_counts.index:
            indexes_class = labels[labels == target_class].index.to_numpy()
            class_idx = np.random.RandomState(seed=42).permutation(indexes_class)[0:labels_counts[target_class]]
            groups[target_class] = data.loc[class_idx.tolist()]

        logger.info("Starting the generation of visualizations.")

        from tqdm import tqdm
        for target in tqdm(groups):
            filepath = f"{output}/{target}.png"
            if not self._skip_existing or not file_exists(filepath):
                self.transform_one(groups[target], filepath)

        logger.info("Generation finished.")

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        if self._bounds is None:
            raise SyntaxError('Method ::fit must called before ::transform.')
        super().transform(data, labels, output)



