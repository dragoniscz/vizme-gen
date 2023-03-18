import pandas as pd

from . import DatasetPipeline

FILE_PATH = "data/wisconsin_breast_cancer.csv"
CLASSES = [
    'M',
    'B'
]


class BreastCancerDatasetPipeline(DatasetPipeline):
    def __init__(self, drop_duplicates=True, filter_classes=('M', 'B')):
        super().__init__()
        self.drop_duplicates = drop_duplicates
        self.filter_classes = filter_classes

    def _load(self):
        return pd.read_csv(FILE_PATH)

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype({
            'diagnosis': 'category',
        })
        return data

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        # drop id and last empty column
        return data[data.columns.difference(['id'])]

    def target(self) -> str:
        return 'diagnosis'
