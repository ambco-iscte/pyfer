import os
import shutil
import matplotlib.pyplot as plt

from random import shuffle

import cv2 as cv
import numpy as np
import pandas as pd
import yaml

from model.recogniser.tensorutils import encode


def parse(pixel_data: str, shape: (int, int)) -> np.ndarray:
    """
    Parses a FER-2013-formatted pixel data string to an RGB image.
    :param pixel_data: Pixel data; Space-separated pixel luminance values in row-major form.
    :param shape: The desired image shape.
    :return: A grayscale image of the desired shape.
    """
    pixels: list[(int, int, int)] = []
    for lum in pixel_data.split():
        pixels.append(int(lum))
    return cv.cvtColor(np.asarray(pixels, dtype=np.uint8).reshape(shape), cv.COLOR_GRAY2RGB)


def parse_greyscale(pixel_data: str, shape: (int, int)) -> np.ndarray:
    """
    Parses a FER-2013-formatted pixel data string to a Grayscale image.
    :param pixel_data: Pixel data; Space-separated pixel luminance values in row-major form.
    :param shape: The desired image shape.
    :return: A grayscale image of the desired shape.
    """

    pixels: list[int] = []
    for lum in pixel_data.split():
        pixels.append(int(lum))
    image = np.array(pixels * 255, dtype=np.uint8).reshape(shape)
    return image


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
        csv_extended: str,
        config: str,
        train_split: float,
        val_split: float,
        output: str = None,
        plot_balance: bool = False
) -> (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[int]], list[list[int]], list[list[int]]):
    """
    Loads the FER-2013 dataset from a CSV file.
    :param csv: The path to the CSV file containing FER image data.
    :param csv_extended: The path to the CSV file containing FER+ image data.
    :param config: A YAML file describing the training, validation, and testing dataset folders, as well as mapping
        each emotion class ID to its name.
    :param train_split: The % (0-1) of data to use for the training set.
    :param val_split: The % (0-1) of testing data (after splitting training data) to use for the validation set.
    :param output: Optional; if specified, images are saved to their folders as given in the configuration file.
    :param plot_balance: Optional; if True, bar plot of category distribution is shown.
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    fer: pd.DataFrame = pd.read_csv(csv)
    fer_plus: pd.DataFrame = pd.read_csv(csv_extended)

    config = yaml.load(open(config), yaml.CLoader)
    emotions = config['emotions']

    # Iterate through all the rows in the CSV and parse their data
    images: list[(np.ndarray, str)] = []
    for (index, row) in fer.iterrows():
        image = parse(row.pixels, (48, 48, 1))

        fer_plus_votes = fer_plus.iloc[index, -10:].tolist()
        emotion = emotions[np.argmax(fer_plus_votes)]

        images.append((image, emotion))

    assert 0 <= train_split <= 1
    assert 0 <= val_split <= 1

    # Shuffle for randomness in train-val-test splitting
    shuffle(images)

    # Check dataset balance
    if plot_balance:
        value_counts = {
            category: len([1 for (_, emotion) in images if emotion == category])
            for category in emotions.values()
        }
        plt.title('Emotion Distribution for FER+ Dataset')
        x = list(value_counts.keys())
        y = list(value_counts.values())
        plt.bar(x, y, color='#546f7c')
        plt.xticks(fontsize=20)
        frame = plt.gca()
        frame.axes.yaxis.set_visible(False)
        for i in range(len(x)):
            plt.text(i, y[i] + 80, y[i], fontsize=20, ha='center')

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

    y_train = [encode(emotion, emotions) for (_, emotion) in train]
    y_val = [encode(emotion, emotions) for (_, emotion) in val]
    y_test = [encode(emotion, emotions) for (_, emotion) in test]

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
    load(
        'fer2013.csv',
        'fer2013new.csv',
        'config.yaml',
        0.8,
        0.1,
        plot_balance=True
    )
    plt.tight_layout()
    plt.show()
