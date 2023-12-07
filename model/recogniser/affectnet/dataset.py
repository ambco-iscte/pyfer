import os
import random
import time
import yaml

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from model.recogniser.tensorutils import encode


def prepare(config: str, split: float):
    return  # We don't want to accidentally run this again

    assert 0 <= split <= 1

    cfg = yaml.load(open(config), yaml.CLoader)
    train = os.path.join('data', cfg["train"])
    train_images = os.path.join(train, 'images')
    test = os.path.join('data', cfg["test"])

    files: list[str] = os.listdir(train_images)
    folder_size = len(files)
    test_size: int = int(split * folder_size)
    print(f'Moving {round(100 * split, 2)}% = {test_size} files from training to testing set.')

    start_time = time.time()
    for i in random.sample(range(folder_size), test_size):
        name = ''.join(os.path.splitext(files[i])[:-1])

        image = os.path.join(train, 'images', f'{name}.jpg')
        annotation = os.path.join(train, 'annotations', f'{name}_exp.npy')

        os.rename(image, os.path.join(test, 'images', f'{name}.jpg'))
        os.rename(annotation, os.path.join(test, 'annotations', f'{name}_exp.npy'))

    print(f'Done! Took {time.time() - start_time} seconds.')


def load_affectnet(
        config: str,
        data: str,
        balanced: bool = False
) -> (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[int]], list[list[int]], list[list[int]]):
    config = yaml.load(open(config), yaml.CLoader)
    emotions: dict[int, str] = config['emotions']

    def load(folder: str) -> dict[str, list[np.ndarray]]:  # dict[Emotion, List[Image]]
        d = defaultdict(list)

        images_folder = os.path.join(folder, 'images')
        annotations_folder = os.path.join(folder, 'annotations')
        image_filenames = os.listdir(images_folder)

        for index in range(len(image_filenames)):
            name = ''.join(os.path.splitext(image_filenames[index])[:-1])
            img = cv.cvtColor(cv.imread(os.path.join(images_folder, f'{name}.jpg')), cv.COLOR_BGR2RGB)
            annotation = os.path.join(annotations_folder, f'{name}_exp.npy')
            try:
                em = emotions[int(np.load(annotation))]
            except FileNotFoundError:
                continue
            d[em].append(img)

        return d

    train_folder = os.path.join(data, config["train"])
    val_folder = os.path.join(data, config["val"])
    test_folder = os.path.join(data, config["test"])

    train = load(train_folder)
    val = load(val_folder)
    test = load(test_folder)

    x_train: list[np.ndarray] = []
    x_val: list[np.ndarray] = []
    x_test: list[np.ndarray] = []
    y_train = []
    y_val = []
    y_test = []

    if balanced:
        smallest = min([len(x) for x in train.values()])
        for emotion in emotions.values():
            train[emotion] = random.sample(train[emotion], smallest)

    for emotion in emotions.values():
        train_images = train[emotion]
        val_images = val[emotion]
        test_images = test[emotion]

        x_train.extend(train_images)
        x_val.extend(val_images)
        x_test.extend(test_images)

        y_train.extend([emotion] * len(train_images))
        y_val.extend([emotion] * len(val_images))
        y_test.extend([emotion] * len(test_images))

    # One-hot encoding
    y_train = [encode(emotion, emotions) for emotion in y_train]
    y_val = [encode(emotion, emotions) for emotion in y_val]
    y_test = [encode(emotion, emotions) for emotion in y_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
    prepare('config.yaml', 0.1)
