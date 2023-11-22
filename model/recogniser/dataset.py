import os
import shutil
from random import shuffle

import cv2 as cv
import numpy as np
import pandas as pd
import yaml


def parse(pixel_data: str, shape: (int, int)) -> np.ndarray:
    """
    Parses a FER-2013-formatted pixel data string to a Grayscale image.
    :param pixel_data: Pixel data; Space-separated pixel luminance values in row-major form.
    :param shape: The desired image shape.
    :return: A grayscale image of the desired shape.
    """
    data = pixel_data.split()
    pixels: list[(int, int, int)] = []
    for lum in data:
        pixels.append(int(lum))
    return np.asarray(pixels, dtype=np.uint8).reshape(shape)


def write(images: list[(np.ndarray, str)], folder: str):
    """
    Writes a list of emotion-annotated images into a folder.
    :param images: A list of (image, emotion) pairs.
    :param folder: The desired output folder.
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for i, (image, emotion) in enumerate(images):
        cv.imwrite(f'{folder}/{i}-{emotion}.jpg', cv.cvtColor(image, cv.COLOR_GRAY2RGB))


def load(
        csv: str,
        config: str,
        train_split: float,
        val_split: float,
        output: str = None
) -> (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]):
    """
    Loads the FER-2013 dataset from a CSV file.
    :param csv: The path to the CSV file containing image data.
    :param config: A YAML file describing the training, validation, and testing dataset folders, as well as mapping each emotion class ID to its name.
    :param train_split: The % (0-1) of data to use for the training set.
    :param val_split: The % (0-1) of testing data (after splitting training data) to use for the validation set.
    :param output: Optional; if specified, images are saved to their folders as given in the configuration file.
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    fer: pd.DataFrame = pd.read_csv(csv)
    config = yaml.load(open(config), yaml.CLoader)
    emotions = config['emotions']

    # Iterate through all the rows in the CSV and parse their data
    images: list[(np.ndarray, str)] = []
    for (index, row) in fer.iterrows():
        image = parse(row.pixels, (48, 48, 1))
        emotion = emotions[row.emotion]
        images.append((image, emotion))

    assert 0 <= train_split <= 1
    assert 0 <= val_split <= 1

    # Shuffle for randomness in train-val-test splitting
    shuffle(images)

    # Get splitting indices
    train_split_index = int(len(images) * train_split)
    val_split_index = train_split_index + int(len(images) * val_split)

    # Split dataset into training, validation, and testing sets
    train: list[(np.ndarray, str)] = images[:train_split_index]
    val: list[(np.ndarray, str)] = images[train_split_index:val_split_index]
    test: list[(np.ndarray, str)] = images[val_split_index:]

    print("Training Set Size      =", len(train))
    print("Validation Set Size    =", len(val))
    print("Testing Set Size       =", len(test))

    # Write each set to its own folder
    if output is not None:
        write(train, f'{output}/{config["train"]}')
        write(val, f'{output}/{config["val"]}')
        write(test, f'{output}/{config["test"]}')

    x_train = [img for (img, _) in train]
    x_val = [img for (img, _) in val]
    x_test = [img for (img, _) in test]
    y_train = [emotion for (_, emotion) in train]
    y_val = [emotion for (_, emotion) in train]
    y_test = [emotion for (_, emotion) in train]

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
    load('fer2013.csv', 'config.yaml', 0.8, 0.1, 'images')
