import datetime
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import yaml
from keras import layers
from keras.src.applications import ResNet50V2
from keras.src.callbacks import EarlyStopping

from model.recogniser.affectnet.dataset import load_affectnet
from model.recogniser.fer.dataset import load as load_fer
from model.recogniser.tensorutils import plot_metric, plot_confusion_matrix, full_evaluate, visualize

matplotlib.use('TkAgg')

# Disable TensorFlow warnings
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
tf.get_logger().setLevel('INFO')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def execute(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        labels: list[str],
        with_pretrained_model: bool,
        img_width: int,
        img_height: int,
        finetuning: int,
        patience: int,
        model_name: str,
        learning_rate: float,
        max_epochs: int,
        batch_size: int
):
    """

    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param x_test:
    :param y_test:
    :param labels: List of unique target labels.
    :param with_pretrained_model: True if transfer learning from ResNet50V2 should be used; False, otherwise.
    :param img_width: Input image width.
    :param img_height: Input image height.
    :param finetuning: Amount of pretrained model layers to finetune; Only applicable if with_pretrained_model is True.
    :param patience:
    :param model_name: Neural network model name.
    :param learning_rate: Learning rate - [0,1].
    :param max_epochs: Maximum number of allowed epochs.
    :param batch_size: Batch size.
    :return:
    """
    num_classes = len(labels)
    print(f"\nUsing {num_classes} classes:\n\t" + '\n\t'.join(labels))

    # Print shapes
    print()
    print(f'Training data shape     = {x_train.shape}')
    print(f'Training labels shape   = {y_train.shape}')
    print(f'Validation data shape   = {x_val.shape}')
    print(f'Validation labels shape = {y_val.shape}')
    print(f'Test data shape         = {x_test.shape}')
    print(f'Test labels shape       = {y_test.shape}\n')

    if with_pretrained_model:
        convolutional_model = ResNet50V2(input_shape=(img_width, img_height, 3), include_top=False)
        convolutional_model.trainable = False

        for layer in convolutional_model.layers[-finetuning:]:
            layer.trainable = True
    else:
        convolutional_model = tf.keras.Sequential([
            layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                input_shape=(img_width, img_height, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.4),
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.4),
            layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])

    # This callback will try to minimise the validation loss, and will stop the learning process if
    # its value has stagnated for at least 5 consecutive epochs.
    # We can check how many epochs were run with len(history.history['loss']).
    callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    # CNN using categorical classification with softmax activation
    model_title = f"{model_name}\nPyFER Emotion Classifier"
    model = tf.keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.2),
        layers.Rescaling(1. / 255),

        convolutional_model,

        layers.Dropout(0.5),
        layers.Flatten(),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ], name=model_name)

    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),  # Calculates entropy for multi-class classification problems
        optimizer=tf.optimizers.AdamW(learning_rate=learning_rate),  # Variation of Gradient Descent

        # Automatically uses Categorical Accuracy.
        # See here: https://stackoverflow.com/questions/55828344/how-does-tensorflow-calculate-the-accuracy-of-model
        metrics=['accuracy']
    )

    model.summary()
    print()

    # Evaluate the CNN
    history, metric_values, time_value, epochs, best_epoch, conf_matrix = full_evaluate(
        model,
        [callback],
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        True,
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    model.save(f'./saved_models/{model_name}', save_format='keras')

    """
        ---------------- Show Results -----------------
    """
    # Round running time to 2 decimal places to avoid unnecessary precision
    time_value = str(datetime.timedelta(seconds=round(time_value)))

    # Accuracy
    figa, axa = plt.subplots()
    plot_metric(
        axa, 'accuracy', history,
        f'{model_title}\nAccuracy\nProgression over {epochs} epochs\n\nRun Time = {time_value}s',
        'Epoch', 'Accuracy', best_epoch
    )
    plt.tight_layout()
    figa.canvas.manager.set_window_title(f'PyFER - Progression of Accuracy over Training for {model_name}')

    # Loss
    figl, axl = plt.subplots()
    plot_metric(
        axl, 'loss', history,
        f'{model_title}\nLoss\nProgression over {epochs} epochs\n\nRun Time = {time_value}s\n',
        'Epoch', 'Loss', best_epoch
    )
    plt.tight_layout()
    figl.canvas.manager.set_window_title(f'PyFER - Progression of Loss over Training for {model_name}')

    # Confusion Matrices
    figm, axm = plt.subplots()
    plot_confusion_matrix(
        conf_matrix,
        axm,
        f"{model_title}\n\n{time_value} over {epochs} epochs",
        labels,
        ['loss', 'accuracy'],
        metric_values
    )
    figm.canvas.manager.set_window_title(f'PyFER - Confusion Matrices for {model_name}')

    plt.tight_layout()


def fer():
    # Configuration file
    config = 'fer/config.yaml'

    # Get data
    x_train, x_val, x_test, y_train, y_val, y_test = load_fer(
        'fer/fer2013.csv',
        'fer/fer2013new.csv',
        config,
        0.8,
        0.1
    )
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Get configuration
    config = yaml.load(open(config), yaml.CLoader)
    emotions = config['emotions']

    # Visualize a few images from the original training set
    visualize(5, 5, x_train, y_train, emotions)

    # Get image width and height
    print(x_train[0].shape)
    img_width, img_height, _ = x_train[0].shape

    # Execute model
    execute(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        labels=emotions.values(),
        with_pretrained_model=True,
        img_width=img_width,
        img_height=img_height,
        finetuning=6,
        patience=10,
        model_name='WithTransferLearningFromResNet50V2',
        learning_rate=0.001,
        max_epochs=100,
        batch_size=100
    )


def affectnet():
    # Configuration file
    config = 'affectnet/config.yaml'

    # Get data
    start_time = time.time()
    print(f'Loading AffectNet dataset...')
    x_train, x_val, x_test, y_train, y_val, y_test = load_affectnet(
        config,
        'affectnet/data',
        balanced=True
    )
    print(f'Done! Took {time.time() - start_time} seconds.\n')

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Get configuration
    config = yaml.load(open(config), yaml.CLoader)
    emotions = config['emotions']

    # Visualize a few images from the original training set
    visualize(5, 5, x_train, y_train, emotions)

    # Get image width and height
    print(x_train[0].shape)
    img_width, img_height, _ = x_train[0].shape

    # Execute model
    execute(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        labels=emotions.values(),
        with_pretrained_model=True,
        img_width=img_width,
        img_height=img_height,
        finetuning=6,
        patience=10,
        model_name='AffectNetWithTransferLearningFromResNet50V2',
        learning_rate=0.001,
        max_epochs=100,
        batch_size=250
    )


if __name__ == "__main__":
    # fer()
    affectnet()
    plt.show()
