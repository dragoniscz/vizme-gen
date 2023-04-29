import abc
import argparse
import json
import os
import pandas as pd
import numpy as np
from loguru import logger

from util.dataset import get_samples_for_labels


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
        self._n_samples = np.inf
        self._parameters = {}

    def skip_existing(self, skip_existing: bool = True) -> None:
        self._skip_existing = skip_existing

    def n_samples(self, n_samples = np.inf) -> None:
        self._n_samples = n_samples

    def setup(self, parameters = None) -> None:
        self._parameters = parameters if parameters is not None else {}

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform_one(self, data: pd.DataFrame, output: str) -> None:
        pass

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        logger.info("Creating output dir: {0}", output)
        create_folder(output)

        (samples, samples_labels) = get_samples_for_labels(self._n_samples, data, labels)

        logger.info("Starting the generation of visualizations.")

        from tqdm import tqdm
        for i in tqdm(range(0, samples.shape[0])):
        #for key, label in tqdm(zip(data.index, labels), total=data.shape[0]):
            label = samples_labels.iloc[i]
            sample = samples.iloc[i]

            directory = f"{output}/{label}"
            if not os.path.isdir(directory):
                create_folder(directory)
            filepath = f"{directory}/{sample.name}.png"
            if not self._skip_existing or not file_exists(filepath):
                self.transform_one(sample, filepath)

        logger.info("Generation finished.")

    def transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        logger.info("Creating output dir: {0}", output)
        create_folder(output)

        labels_counts = labels.value_counts().apply(lambda x: x if x <= self._n_samples else self._n_samples)
        total_samples = labels_counts.sum()
        logger.info("Found {0} target classes, {1} samples at total.", labels_counts.index.shape[0], total_samples)

        groups = {}
        for target_class in labels_counts.index:
            indexes_class = labels[labels == target_class].index.to_numpy()
            class_idx = np.random.RandomState(seed=42).permutation(indexes_class)[0:labels_counts[target_class]]
            if grouping == 'avg':
                groups[target_class] = data.loc[class_idx.tolist()].mean()
            elif grouping == 'med':
                groups[target_class] = data.loc[class_idx.tolist()].median()
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

    choices = {
        'blobs': BlobsVisualizationPipeline(),
        'radial': RadialPlotsVisualizationPipeline(),
        'SOM': SOMVisualizationPipeline(),
        'parallel': ParallelCoordinatesVisualizationPipeline(),
    }

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    # noinspection PyMethodOverriding
    def __call__(self, parser, namespace, values, option_strings):
        setattr(namespace, self.dest, Parse.choices[values])
