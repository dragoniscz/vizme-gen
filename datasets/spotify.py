import pandas as pd

from . import DatasetPipeline

FILE_PATH = "data/SpotifyFeatures.csv"
CLASSES = [
    'Movie',
    'A Capella',
    'Alternative',
    'Country',
    'Dance',
    'Electronic',
    'Anime',
    'Folk',
    'Blues',
    'R&B',
    'Opera',
    'Hip-Hop',
    "Children's Music",
    'Rap',
    'Indie',
    'Classical',
    'Pop',
    'Reggae',
    'Reggaeton',
    'Jazz',
    'Rock',
    'Ska',
    'Comedy',
    'Soul',
    'Soundtrack',
    'World',
]


class SpotifyDatasetPipeline(DatasetPipeline):
    def __init__(self, drop_duplicates=True, filter_classes=('Classical', 'Dance', 'Rock', 'Anime')):
        super().__init__()
        self.drop_duplicates = drop_duplicates
        self.filter_classes = filter_classes

    def _load(self):
        return pd.read_csv(FILE_PATH) \
            .replace({'genre': {'Children’s Music': 'Children\'s Music'}})

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.drop_duplicates:
            data = data.drop_duplicates(subset=['track_id'], keep=False)
        if self.filter_classes is not None:
            data = data[data['genre'].isin(self.filter_classes)]
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype({
            'track_id': 'string',
            'genre': 'category',
            'key': 'category',
        }).replace({
            'mode': {'Major': 1, 'Minor': 0},
            'time_signature': {'0/4': 0, '1/4': 1, '2/4': 2, '3/4': 3, '4/4': 4, '5/4': 5},
        })

        data['key'] = pd.factorize(data['key'], sort=True)[0]
        return data

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[data.columns.difference(['artist_name', 'track_name', 'popularity'])] \
            .set_index('track_id')

    def target(self) -> str:
        return 'genre'
