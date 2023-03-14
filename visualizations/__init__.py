import abc
import argparse

import numpy as np
import pandas as pd


def create_folder(directory: str) -> None:
    try:
        import os
        os.mkdir(directory)
    except FileExistsError as _:
        pass


def file_exists(filepath: str) -> bool:
    import os
    return os.path.exists(filepath)


class VisualizationPipeline(abc.ABC):
    def __init__(self, skip_existing: bool = False):
        self._skip_existing = skip_existing

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform_one(self, data: pd.DataFrame, output: str) -> None:
        pass

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        create_folder(output)
        from tqdm import tqdm
        for key, label in tqdm(zip(data.index, labels), total=data.shape[0]):
            directory = f"{output}/{label}"
            create_folder(directory)
            filepath = f"{directory}/{key}.png"
            if not self._skip_existing or not file_exists(filepath):
                self.transform_one(data.loc[key], filepath)

    def fit_transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        self.fit(data, labels)
        return self.transform(data, labels, output)


class Parse(argparse.Action):
    from .blobs import BlobsVisualizationPipeline

    choices = {
        'blobs': BlobsVisualizationPipeline(),
    }

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    # noinspection PyMethodOverriding
    def __call__(self, parser, namespace, values, option_strings):
        setattr(namespace, self.dest, Parse.choices[values])