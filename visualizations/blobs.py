import json
import numpy as np
import pandas as pd

from . import VisualizationPipeline


class BlobsVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()
        self._coords = None
        self._means = None
        self._quantize = None
        self._r_min = 0.0
        self._r_max = 0.2

    def setup(self, parameters = None) -> None:
        super().setup(parameters)
        if self._parameters['r_min'] is not None:
            self._r_min = float(self._parameters['r_min'])
        if self._parameters['r_max']:
            self._r_max = float(self._parameters['r_max'])

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        data = data[data.columns.difference(data.columns[data.isnull().any(axis=0)].tolist())]

        correlation = np.corrcoef(data, rowvar=False)
        assert correlation.shape[0] == len(data.columns) and correlation.shape[1] == len(data.columns)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=0, perplexity=5)
        coords = tsne.fit_transform(np.abs((correlation - 1) / 2))
        coords = (coords - coords.min()) / (coords.max() - coords.min())

        self._coords = pd.DataFrame(coords, index=data.columns)

        self._means = np.mean(data, axis=0)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        self._quantize = make_pipeline(QuantileTransformer(n_quantiles=25)).fit(data.to_numpy())

    def transform_one(self, data: pd.Series, output: str) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import cm

        qd = self._quantize.transform([data])[0]

        fig, ax = plt.subplots()
        for idx, key in enumerate(data.index):
            ax.add_patch(plt.Circle(self._coords.loc[key], radius=self._r_min + (self._r_max - self._r_min) * data[key], alpha=0.5, color=cm.seismic(qd[idx])))
        ax.axis('off')
        fig.set_size_inches(4, 4)
        plt.savefig(output, pad_inches=0, bbox_inches='tight', transparent=False)
        plt.close(fig)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        data = data[data.columns.difference(data.columns[data.isnull().any(axis=0)].tolist())]

        if self._coords is None:
            raise SyntaxError('Method ::fit must called before ::transform.')
        super().transform(data, labels, output)

    def transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        data = data[data.columns.difference(data.columns[data.isnull().any(axis=0)].tolist())]

        if self._coords is None:
            raise SyntaxError('Method ::fit must called before ::transform.')
        super().transform_group(data, labels, output, grouping)



