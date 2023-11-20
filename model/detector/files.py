import os
from math import floor
from random import shuffle

import cv2 as cv
import numpy as np

from typing import Sequence

import pandas as pd

from bounds import BoundingBox


# PyFER - Group 02
# Afonso Caniço     92494
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


# The raw annotations contain the diagonal points of the bounding box.
# YOLOv8 expects the bounding box information in the form of centre coordinates,
# width and height of the bounding box.
def create_labels(images_folder: str, bounds_path: str, output_folder: str):
    bounds_data = pd.read_csv(bounds_path)

    for filename in os.listdir(images_folder):
        image_labels = ""

        image_name = os.path.splitext(filename)[:-1]
        box_entries = bounds_data.loc[bounds_data['image_name'] == filename]
        for index, entry in box_entries.iterrows():
            # boxes.append(BoundingBox(entry.image_name, (entry.x0, entry.y0), (entry.x1, entry.y1), 1.0))
            x_centre = 0.5 * (entry.x0 + entry.x1)
            y_centre = 0.5 * (entry.y0 + entry.y1)
            bb_width = entry.x1 - entry.x0
            bb_height = entry.y1 - entry.y0
            # Also the dimensions of bounding box are to be normalised
            # with respect to image width and height
            x_centre_scaled = x_centre / entry.width
            y_centre_scaled = y_centre / entry.height
            width_scaled = bb_width / entry.width
            height_scaled = bb_height / entry.height

            """
                Creating a text file for every image with the bounding box information in correct format. 
                The correct format for each bounding box is as follows:
                
                class_id   x_centre   y_centre   width   height
            """

            if len(image_labels) != 0:
                image_labels + '\n'
            image_labels = image_labels + "0" + " " + str(x_centre_scaled) + " " + str(y_centre_scaled) + " " + str(
                width_scaled) + " " + str(height_scaled) + "\n"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        f = open(output_folder + "/" + "".join(image_name) + ".txt", 'w')
        f.writelines(image_labels)
        f.close()


def shuffle_and_split_files(images_folder, train_split=0.8, val_split=0.1):

    """
    # function to move files from source to detination
    def move_files(data_list, source_path, destination_path):
        i = 0
        for file in data_list:
            filepath = os.path.join(source_path, file)
            dest_path = os.path.join(data_path, destination_path)
            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)
            shutil.move(filepath, dest_path)
            i = i + 1
        print("Number of files transferred:", i)
    """

    filenames = os.listdir(images_folder)
    shuffle(filenames)

    train_split_index = floor(len(filenames) * train_split)
    val_split_index = train_split_index + floor(len(filenames) * val_split)
    training = filenames[:train_split_index]
    val = filenames[train_split_index + 1:val_split_index]
    test = filenames[val_split_index + 1:]

    print(train_split_index, val_split_index)
