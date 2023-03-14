import argparse
import pandas as pd

from vizme.preprocessing import normalize, quantize


def main(args):
    try:
        dataset = args['datasets'].data()
        assert isinstance(dataset, pd.DataFrame)

        if args['quant']:
            dataset = quantize(dataset)
        if args['norm']:
            dataset = normalize(dataset)

        args['visualization'].fit_transform(dataset, args['datasets'].labels(), args['output'])
        return 0
    except FileNotFoundError:
        print(f'Cannot load datasets.')
        return 1


if __name__ == '__main__':
    from datasets import Parse as DatasetParse
    from visualizations import Parse as VisualizationParse

    parser = argparse.ArgumentParser(description='Tool to convert data into image.')

    parser.add_argument('datasets',
                        choices=list(DatasetParse.choices.keys()),
                        action=DatasetParse,
                        help='Source datasets.')

    parser.add_argument('--norm', '--normalize', '--normalization',
                        default=False,
                        action='store_true',
                        help='Normalization of features.')

    parser.add_argument('--quant', '--quantize', '--quantization',
                        default=False,
                        action='store_true',
                        help='Quantization of features.')

    parser.add_argument('visualization',
                        choices=list(VisualizationParse.choices.keys()),
                        action=VisualizationParse,
                        help='Visualization system.')

    parser.add_argument('output',
                        type=str,
                        help='Directory where visualization should be saved.')

    exit(main(vars(parser.parse_args())))
