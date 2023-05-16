import abc
import argparse
import os
import pandas as pd
import numpy as np
from loguru import logger


def create_folder(directory: str) -> None:
    try:
        import os
        os.makedirs(directory)
    except FileExistsError as _:
        pass


def file_exists(filepath: str) -> bool:
    import os
    return os.path.exists(filepath)


class VisualizationPipeline(abc.ABC):
    def __init__(self):
        self._skip_existing = False
        self._parameters = {}

    def skip_existing(self, skip_existing: bool = True) -> None:
        self._skip_existing = skip_existing

    def setup(self, parameters = None) -> None:
        self._parameters = parameters if parameters is not None else {}

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform_one(self, data: pd.Series, output: str) -> None:
        pass

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        assert data.shape[0] == labels.shape[0]

        logger.info("Creating output dir: {0}", output)
        create_folder(output)

        logger.info("Starting the generation of visualizations.")

        from tqdm import tqdm
        for i in tqdm(range(0, data.shape[0])):
            label = labels.iloc[i]
            sample = data.iloc[i]

            directory = f"{output}/{label}"
            if not os.path.isdir(directory):
                create_folder(directory)
            filepath = f"{directory}/{sample.name}.png"
            if not self._skip_existing or not file_exists(filepath):
                self.transform_one(sample, filepath)

        logger.info("Generation finished.")

    def transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        assert data.shape[0] == labels.shape[0]

        logger.info("Creating output dir: {0}", output)
        create_folder(output)

        labels_counts = labels.value_counts()

        groups = {}
        for target_class in labels_counts.index:
            indexes_class = labels[labels == target_class].index.to_numpy()
            if grouping == 'avg':
                groups[target_class] = data.loc[indexes_class].mean()
            elif grouping == 'med':
                groups[target_class] = data.loc[indexes_class].median()
            else:
                raise ValueError(f"Unsupported grouping '{grouping}'.")

        logger.info("Starting the generation of visualizations.")

        from tqdm import tqdm
        for target in tqdm(groups):
            filepath = f"{output}/{target}.png"
            if not self._skip_existing or not file_exists(filepath):
                self.transform_one(groups[target], filepath)

        logger.info("Generation finished.")

    def fit_transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        self.fit(data, labels)
        return self.transform_group(data, labels, output, grouping)

    def fit_transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        self.fit(data, labels)
        return self.transform(data, labels, output)


class Parse(argparse.Action):
    from .blobs import BlobsVisualizationPipeline
    from .radial_plots import RadialPlotsVisualizationPipeline
    from .SOM import SOMVisualizationPipeline
    from .parallel_coordinates import ParallelCoordinatesVisualizationPipeline
    from .table import TableVisualizationPipeline

    choices = {
        'blobs': BlobsVisualizationPipeline(),
        'radial': RadialPlotsVisualizationPipeline(),
        'SOM': SOMVisualizationPipeline(),
        'parallel': ParallelCoordinatesVisualizationPipeline(),
        'table': TableVisualizationPipeline(),
    }

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    # noinspection PyMethodOverriding
    def __call__(self, parser, namespace, values, option_strings):
        setattr(namespace, self.dest, Parse.choices[values])
