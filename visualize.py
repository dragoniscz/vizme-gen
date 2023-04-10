import argparse
import pandas as pd
import json
import numpy as np
from loguru import logger

from vizme.preprocessing import normalize, quantize


def main(args):
    try:
        dataset = args['datasets'].data()
        assert isinstance(dataset, pd.DataFrame)

        if args['quant']:
            dataset = quantize(dataset)
        if args['norm']:
            dataset = normalize(dataset)

        if args['skip']:
            args['visualization'].skip_existing()
        if args['n_samples'] is not None:
            args['visualization'].n_samples(args['n_samples'])
        if args['parameters'] is not None:
            args['visualization'].setup(args['parameters'])

        args['visualization'].fit_transform(dataset, args['datasets'].labels(), args['output'])
        return 0
    except FileNotFoundError as err:
        logger.error(f'Cannot load datasets.')
        logger.debug(err)
        return 1


if __name__ == '__main__':
    from datasets import Parse as DatasetParse
    from visualizations import Parse as VisualizationParse

    parser = argparse.ArgumentParser(description='Tool to convert data into image.')

    parser.add_argument('--skip', '--skip-existing',
                        default=False,
                        action='store_true',
                        help='Skip existing files (do not generate them again)')

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

    parser.add_argument('--n_samples',
                        default=np.inf,
                        type=np.int32,
                        required=False,
                        help='Number of samples of each target class. If not provided, all samples will be generated.')

    parser.add_argument('-p', '--parameters',
                        default=False,
                        type=json.loads,
                        help='The parameters of the visualization.')

    exit(main(vars(parser.parse_args())))
