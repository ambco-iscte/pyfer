import os

import cv2 as cv
import numpy as np

from typing import Sequence

import pandas as pd

from bounds import BoundingBox

# PyFER - Group 02
# Afonso Caniço     92494
# Gustavo Ferreira  92888
# Samuel Correia    92619


def load_images(images_folder: str, bounds_path: str) -> dict[str, (np.ndarray, Sequence[BoundingBox])]:
    """
    Loads images and their respective coordinates.
    :param images_folder: The folder containing the images and their corresponding annotation files.
    :param bounds_path: The .csv file containing the data relative to the coordinates of the bounding boxes of faces.
    :return: A dictionary associating each file to its image and list of bounding boxes.
    """
    images = {}

    bounds_data = pd.read_csv(bounds_path)

    for filename in os.listdir(images_folder):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue
        image_path = os.path.join(images_folder, filename)
        image = cv.imread(image_path)

        boxes = list()
        box_entries = bounds_data.loc[bounds_data['image_name'] == filename]
        for index, entry in box_entries.iterrows():
            boxes.append(BoundingBox(entry.image_name, (entry.x0, entry.y0), (entry.x1, entry.y1), 1.0))

        images[filename] = (image, boxes)

    return images
