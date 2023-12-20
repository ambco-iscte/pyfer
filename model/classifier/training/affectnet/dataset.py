import os
import random
import time

import cv2
import keras
import yaml

import numpy as np
import tensorflow as tf

# Builds training and testing sets from AffectNet's original big training set
def prepare(config: str, split: float):
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


# Prepares the dataset to be used with Keras's image_dataset_from_directory method.
def prepare_for_tf_dataset(config: str):
    cfg = yaml.load(open(config), yaml.CLoader)
    emotions = cfg['emotions']

    def prep(path: str):
        print(f'Preparing dataset for {path}')
        path_images = os.path.join(path, 'images')
        for img_filename in os.listdir(path_images):
            name = ''.join(os.path.splitext(img_filename)[:-1])

            image = os.path.join(path, 'images', f'{name}.jpg')
            try:
                annotation = os.path.join(path, 'annotations', f'{name}_exp.npy')
                emotion = emotions[int(np.load(annotation))]

                emotion_folder = os.path.join(path, emotion)

                if not os.path.isdir(emotion_folder):
                    os.makedirs(emotion_folder)

                os.rename(image, os.path.join(emotion_folder, f'{name}.jpg'))
            except FileNotFoundError:
                print(f'{image} is not annotated, skipping')

    train = os.path.join('data', cfg["train"])
    test = os.path.join('data', cfg["test"])
    val = os.path.join('data', cfg["val"])

    prep(train)
    prep(test)
    prep(val)


def load_affectnet(
        config: str,
        data: str,
        shape: (int, int) = (224, 224),
        train_samples: int = 32,
        test_samples: int = 32,
        val_samples: int = 32
) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):  # (Train, Test, Validation)
    config = yaml.load(open(config), yaml.CLoader)
    emotions: list[str] = config['emotions'].values()

    img_width, img_height = shape

    train_path = os.path.join(data, config['train'])
    test_path = os.path.join(data, config['test'])
    val_path = os.path.join(data, config['val'])

    print('Loading AffectNet dataset...')
    start_time = time.time()

    train_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=train_samples,
        class_names=emotions,
        shuffle=True,
        color_mode='rgb'
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        test_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=test_samples,
        class_names=emotions,
        shuffle=True,
        color_mode='rgb'
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        val_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=val_samples,
        class_names=emotions,
        shuffle=True,
        color_mode='rgb'
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f'Done! Took {time.time() - start_time} seconds.\n')

    return train_ds, test_ds, val_ds
