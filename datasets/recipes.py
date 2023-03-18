import pandas as pd

from . import DatasetPipeline

FILE_PATH = "data/recipes_nutrients_ingredients.csv"
CLASSES = [
    'Casseroles',
    'Pasta',
    'Roasts',
    'VegetableSalads',
]


class RecipesDatasetPipeline(DatasetPipeline):
    def __init__(self, drop_duplicates=True, filter_classes=('Casseroles', 'Pasta', 'Roasts', 'VegetableSalads')):
        super().__init__()
        self.drop_duplicates = drop_duplicates
        self.filter_classes = filter_classes

    def _load(self):
        return pd.read_csv(FILE_PATH)

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype({
            'category': 'category',
        })
        return data

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        # return data.iloc[:, 23:]
        return data[data.columns.difference(['recipe_name', 'niacin', 'vitaminB6', 'vitaminC', 'sodium',
                                             'carbohydrates', 'sugars', 'calories', 'thiamin', 'caloriesFromFat',
                                             'folate', 'calcium', 'fiber', 'fat', 'iron', 'cholesterol', 'protein',
                                             'vitaminA', 'potassium', 'saturatedFat', 'magnesium'])]

    def target(self) -> str:
        return 'category'
