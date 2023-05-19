import pandas as pd

from . import VisualizationPipeline


class ChernoffFaceVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass


    def transform_one(self, data: pd.Series, output: str) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from ChernoffFace import chernoff_face

        fig = chernoff_face(data=data.to_list(), color_mapper=cm.Pastel1)
        fig.set_size_inches(4, 4)
        plt.savefig(output, pad_inches=0, bbox_inches='tight', transparent=False)
        plt.close(fig)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        super().transform(data, labels, output)

    def transform_group(self, data: pd.DataFrame, labels: pd.DataFrame, output: str, grouping: str) -> None:
        super().transform_group(data, labels, output, grouping)



