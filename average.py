import argparse
import os.path

import numpy as np


def get_directories(directory=None):
    from os import listdir, getcwd

    if directory is None:
        directory = getcwd()

    return [filename for filename in listdir(directory) if os.path.isdir(directory + '/' + filename)]


def get_files(directory=None, extensions=None):
    from os import listdir, getcwd

    if directory is None:
        directory = getcwd()
    if extensions is None:
        extensions = ['png']

    return [directory + '/' + filename for filename in listdir(directory) if filename.split('.')[-1] in extensions]


def merge_files(files):
    from PIL import Image

    width, height = Image.open(files[0]).size
    count = len(files)

    output = np.zeros((height, width, 4), float)
    for file in files:
        image = np.array(Image.open(file), dtype=float)
        output = output + image / count

    return np.array(np.round(output), dtype=np.uint8)


def save_image(image, target='average.png', debug=False):
    from PIL import Image

    output = Image.fromarray(image, mode='RGBA')
    output.save(target)
    if debug:
        output.show()


def process(directory, category):
    files = get_files(directory + "/" + category)
    if len(files) != 0:
        output = merge_files(files)
        save_image(output, directory + "/average_" + category + '.png')

        gy, gx = np.gradient((output[:, :, 0] + output[:, :, 1] + output[:, :, 2]) / 2)
        sharpness = np.average(np.sqrt(gx**2 + gy**2))
        print(category + " " + str(len(files)) + " [" + str(sharpness) + "]")


def main(args):
    folder = args['folder']
    for directory in get_directories(folder):
        process(folder, directory)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to make average of all visualizations.')

    parser.add_argument('folder',
                        type=str,
                        help='Folder with visualizations with subfolder for separate targets.')

    exit(main(vars(parser.parse_args())))

