import argparse
import pandas as pd
import numpy as np
from loguru import logger

from util.dataset import get_samples_for_labels


def main(args):
    try:
        dataset = args['datasets'].data()
        assert isinstance(dataset, pd.DataFrame)
        labels = args['datasets'].labels()

        if args['n_samples']:
            dataset, labels = get_samples_for_labels(args['n_samples'], dataset, labels)

        dataset['_target'] = labels
        dataset.to_csv(args['output'])
        return 0
    except FileNotFoundError as err:
        logger.error(f'Cannot load datasets.')
        logger.debug(err)
        return 1


if __name__ == '__main__':
    from datasets import Parse as DatasetParse

    parser = argparse.ArgumentParser(description='Tool to convert data into image.')

    parser.add_argument('datasets',
                        choices=list(DatasetParse.choices.keys()),
                        action=DatasetParse,
                        help='Source datasets.')

    parser.add_argument('output',
                        type=str,
                        help='Directory where visualization should be saved.')

    parser.add_argument('-s', '--n_samples',
                        default=np.inf,
                        type=np.int32,
                        required=False,
                        help='Number of samples of each target class. If not provided, all samples will be generated.')

    exit(main(vars(parser.parse_args())))
