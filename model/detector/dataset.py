import os
import shutil
import time

import cv2 as cv
import pandas as pd

from random import shuffle

# PyFER - Group 02
# Afonso Caniço     92494
# Samuel Correia    92619


FACES_FILE = "data/faces.csv"
IMAGE_FOLDER = "data/images"
LABEL_FOLDER = "data/labels"
DATASET_FOLDER = "datasets"


class DatasetBuilder:

    def __init__(self, images_folder: str, labels_folder: str):
        self.images_folder = images_folder
        self.labels_folder = labels_folder

    def create_labels(self, bounds_path: str):
        """
        Annotates each image in a given folder with text files containing face bounding box information in YOLO format.
        :param bounds_path: Path to the CSV file containing bounding box information.
        """
        bounds_data = pd.read_csv(bounds_path)

        for filename in os.listdir(self.images_folder):
            image_labels = ""

            image_name = os.path.splitext(filename)[:-1]
            box_entries = bounds_data.loc[bounds_data['image_name'] == filename]

            for index, entry in box_entries.iterrows():
                # Bounding box centre coordinates
                x_centre = 0.5 * (entry.x0 + entry.x1)
                y_centre = 0.5 * (entry.y0 + entry.y1)

                # Bounding box dimensions
                bb_width = entry.x1 - entry.x0
                bb_height = entry.y1 - entry.y0

                # Normalise dimensions with respect to image width and height
                x_centre_scaled = x_centre / entry.width
                y_centre_scaled = y_centre / entry.height
                width_scaled = bb_width / entry.width
                height_scaled = bb_height / entry.height

                # Creating a text file for each image with the bounding box information in the following format:
                # class_id x_centre y_centre width height
                if len(image_labels) != 0:
                    image_labels += '\n'
                image_labels += f'0 {x_centre_scaled} {y_centre_scaled} {width_scaled} {height_scaled}'

            if not os.path.isdir(self.labels_folder):
                os.makedirs(self.labels_folder)

            f = open(self.labels_folder + "/" + "".join(image_name) + ".txt", 'w')
            f.writelines(image_labels)
            f.close()

    def build(
            self,
            bounds_path: str,
            output_folder: str,
            train_split: float = 0.8,
            val_split: float = 0.1,
            resized_image_size_x: int = 640,
            resized_image_size_y: int = 640
    ):
        """
        Build training, validation, and testing datasets.
        :param bounds_path: Path to the CSV file containing bounding box information.
        :param output_folder: Path where each of the three datasets will be stored (each in its own sub-folder).
        :param train_split: The % (0-1) of data to use for the training set.
        :param val_split: The % (0-1) of testing data (after splitting training data) to use for the validation set.
        :param resized_image_size_x: Normalised image width.
        :param resized_image_size_y: Normalised image height.
        """

        def move_labels_images_and_resize(files_to_move, folder):
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

            for image in files_to_move:
                # Moves labels considering the image name with the extension: .txt
                label_path = os.path.join(self.labels_folder, "".join(os.path.splitext(image)[:-1]) + ".txt")
                shutil.copy(label_path, folder)

                # Move resized version of the images to the output folder.
                image_path = os.path.join(self.images_folder, image)
                img_resized = cv.resize(cv.imread(image_path), (resized_image_size_x, resized_image_size_y))
                output_image_path = os.path.join(folder, image)
                cv.imwrite(output_image_path, img_resized)

        # Create text files with bounding box information
        print('Building bounding box annotation files...')
        self.create_labels(bounds_path)
        print('Done!')

        filenames = os.listdir(self.images_folder)

        print("\nTotal Number of Images =", len(filenames))

        # Shuffle filenames
        shuffle(filenames)

        train_split_index = int(len(filenames) * train_split)
        val_split_index = train_split_index + int(len(filenames) * val_split)

        # Split images between training, validation, and testing sets
        train = filenames[:train_split_index]
        val = filenames[train_split_index:val_split_index]
        test = filenames[val_split_index:]

        print("Training Set Size      =", len(train))
        print("Validation Set Size    =", len(val))
        print("Testing Set Size       =", len(test))

        # Write each dataset to its folder
        print('\nWriting training set...')
        start_time = time.time()
        move_labels_images_and_resize(train, output_folder + "/train")
        print(f'Done! Took {time.time() - start_time} seconds.')

        print('\nWriting training set...')
        start_time = time.time()
        move_labels_images_and_resize(val, output_folder + "/val")
        print(f'Done! Took {time.time() - start_time} seconds.')

        print('\nWriting training set...')
        start_time = time.time()
        move_labels_images_and_resize(test, output_folder + "/test")
        print(f'Done! Took {time.time() - start_time} seconds.')


if __name__ == "__main__":
    DatasetBuilder(IMAGE_FOLDER, LABEL_FOLDER).build(
        FACES_FILE,
        DATASET_FOLDER
    )
