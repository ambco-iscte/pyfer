import datetime
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import yaml
from keras import layers
from keras.src.applications import MobileNetV3Small, ResNet50V2
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from keras.applications.mobilenet import MobileNet

from model.recogniser.dataset import load
from model.recogniser.tensorutils import plot_metric, plot_confusion_matrix, full_evaluate, visualize

matplotlib.use('TkAgg')

# Disable TensorFlow warnings
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
tf.get_logger().setLevel('INFO')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run(model_name: str, config: str, max_epochs: int, batch_size: int, patience: int, learning_rate: float = 0.001):
    # Get data
    x_train, x_val, x_test, y_train, y_val, y_test = load(
        'fer2013.csv',
        config,
        0.8,
        0.1
    )
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Visualize a few images from the original training set
    visualize(7, 7, x_train)

    # Get image width and height
    print(x_train[0].shape)
    img_width, img_height, _ = x_train[0].shape

    # Automatically collect labels and calculate the number of classes
    config = yaml.load(open(config), yaml.CLoader)
    emotions = config['emotions']
    labels = emotions.values()
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

    with_pretrained_model = False
    if with_pretrained_model:
        # pretrained_model = MobileNetV3Small(input_shape=(48, 48, 3), include_top=False)
        convolutional_model = ResNet50V2(input_shape=(48, 48, 3), include_top=False)
        convolutional_model.trainable = False

        for layer in convolutional_model.layers[-3:]:
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

        # TODO: figure out a model (everything we've tried so far sucks)
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

    model.save(f'./saved_models/{model_name}')
    # keras.models.load_model("my_model.keras")

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


def main():
    epochs = 100
    batch_size = 100
    patience = 10

    run("CustomCNN", 'config.yaml', epochs, batch_size, patience)
    plt.show()


if __name__ == "__main__":
    main()
