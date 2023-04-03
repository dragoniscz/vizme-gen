import abc
import argparse
import pandas as pd


class DatasetPipeline(abc.ABC):
    def __init__(self):
        self._cache = None

    @abc.abstractmethod
    def _load(self) -> pd.DataFrame:
        pass

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def process(self) -> pd.DataFrame:
        if self._cache is None:
            data = self._load()
            data = self._filter(data)
            data = self._transform(data)
            data = self._select(data)
            self._cache = data
        return self._cache

    def data(self) -> pd.DataFrame:
        data = self.process()
        return data[data.columns.difference([self.target()])]

    def labels(self) -> pd.DataFrame:
        return self.process()[self.target()]

    @abc.abstractmethod
    def target(self) -> str:
        pass


class Parse(argparse.Action):
    from .spotify import SpotifyDatasetPipeline
    from .recipes import RecipesDatasetPipeline
    from .recipes_weighted import RecipesWeightedDatasetPipeline
    from .breast_cancer import  BreastCancerDatasetPipeline

    choices = {
        'spotify': SpotifyDatasetPipeline(),
        'recipes': RecipesDatasetPipeline(),
        'recipesWeighted': RecipesWeightedDatasetPipeline(),
        'breastCancer': BreastCancerDatasetPipeline(),
    }

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    # noinspection PyMethodOverriding
    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, Parse.choices[values])