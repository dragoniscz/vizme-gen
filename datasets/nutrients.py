import pandas as pd

from . import DatasetPipeline

FILE_PATH = "data/nutrients.csv"
CLASSES = (
    'Dairy and Egg Products',
    'Spices and Herbs',
    'Baby Foods',
    'Fats and Oils',
    'Poultry Products',
    'Soups, Sauces, and Gravies',
    'Sausages and Luncheon Meats',
    'Breakfast Cereals',
    'Fruits and Fruit Juices',
    'Pork Products',
    'Vegetables and Vegetable Products',
    'Nut and Seed Products',
    'Beef Products',
    'Beverages',
    'Finfish and Shellfish Products',
    'Legumes and Legume Products',
    'Lamb, Veal, and Game Products',
    'Baked Products',
    'Sweets',
    'Cereal Grains and Pasta',
    'Fast Foods',
    'Meals, Entrees, and Sidedishes',
    'Snacks',
    'Ethnic Foods',
    'Restaurant Foods',
)


class NutrientsDatasetPipeline(DatasetPipeline):
    def __init__(self, drop_duplicates=True, filter_classes=CLASSES):
        super().__init__()
        self.drop_duplicates = drop_duplicates
        self.filter_classes = filter_classes

    def _load(self):
        return pd.read_csv(FILE_PATH)

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.filter_classes is not None:
            data = data[data['group'].isin(self.filter_classes)]
        if self.drop_duplicates:
            data = data.drop_duplicates(subset=['name'], keep=False)
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype({
            'name': 'string',
            'group': 'category',
        })

        data['name'] = data['name'].str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
        data['group'] = data['group'].str.replace(r'[^a-zA-Z0-9]', '_', regex=True)

        return data

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.set_index('name')

    def target(self) -> str:
        return 'group'
