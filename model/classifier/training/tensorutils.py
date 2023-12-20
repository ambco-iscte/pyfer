import time
from collections import defaultdict

import keras.src.callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from typing import Any

from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def encode(item: str, mapping: dict[int, str]) -> list[int]:
    """
    Applies one-hot encoding to item as pertaining to a dictionary mapping each index to a class.
    :param item: The class to one-hot encode.
    :param mapping: Dictionary mapping each index to a class.
    :return: A one-hot-encoded representation of the item.
    """
    default = [0] * len(mapping.keys())
    for i, e in mapping.items():
        if item == e:
            default[i] = 1
            return default
    return default


def decode(item: list[int] | tf.Tensor, mapping: dict[int, str]) -> str:
    """
    Decodes a one-hot-encoded element according to a given dictionary mapping each index to a class.
    :param item: One-hot-encoded element. (Python list or TensorFlow tensor.)
    :param mapping: Dictionary mapping each index to a class.
    :return:
    """
    if isinstance(item, list):
        return mapping[np.argmax(item)]
    return mapping[tf.math.argmax(item).numpy()]


# Displays a (width * height) grid of plots, each showcasing one element from the provided data.
# Useful since we're using images and this allows us to visualize a bunch of them in a single, neat plot. :)
def visualize(width: int, height: int, x, y, decoding: dict[int, str]):
    fig, ax = plt.subplots(height, width)
    for i in range(height):
        for j in range(width):
            img = x[i * height + j]
            ax[i, j].imshow(img)
            emotion = decoding[np.argmax(y[i * height + j])]
            ax[i, j].set_title(emotion)
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.canvas.manager.set_window_title('PyFER - Input Images')
    plt.suptitle(f'First {width * height} Training Samples')
    plt.tight_layout()
    return fig


# Generates a plot for a trained model's metric.
# Useful for displaying accuracy, loss, etc.
def plot_metric(axis, metric, history, title, x_label, y_label):
    axis.plot(history.history[metric])
    axis.plot(history.history[f'val_{metric}'])
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)
    axis.legend(['Training', 'Validation'])


def plot_confusion_matrix(matrix, axis, title, labels, metrics, metric_values) -> ConfusionMatrixDisplay:
    plt.figure(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=axis, colorbar=False, cmap="Blues", xticks_rotation=45)
    cf_title = f'{title}\n'
    for i, metric in enumerate(metrics):
        cf_title += f"\n{metric.capitalize()} = {round(metric_values[i], 2)}"
    axis.set_title(cf_title)
    return disp

def dataset_weights(ds: tf.data.Dataset) -> dict[int, float]:
    class_weights: dict[int, float] = defaultdict(float)
    total_number_samples = ds.cardinality().numpy()

    labels: set[int] = set()
    occurrences: dict[int, int] = defaultdict(int)

    print('Calculating class weights for Keras Dataset...')

    # Get number of unique labels and number of items for each label
    for (x, y) in ds:
        for label_tensor in y:
            label = tf.math.argmax(label_tensor).numpy()
            occurrences[label] += 1
            labels.add(label)

    n_classes = len(labels)
    for label_index in labels:
        # w_j= total_number_samples / (n_classes * n_samples_j)
        class_weights[label_index] = total_number_samples / (n_classes * occurrences[label_index])

    print('\tDone!')
    return class_weights


def numpy_weights(y: np.ndarray) -> dict[int, float]:
    class_weights: dict[int, float] = defaultdict(float)

    total_number_samples = len(y)
    labels: set[int] = set()
    occurrences: dict[int, int] = defaultdict(int)

    print('Calculating class weights for Numpy array...')

    # Get number of unique labels and number of items for each label
    for label in y:
        label_index = tf.math.argmax(label).numpy()
        occurrences[label_index] += 1
        labels.add(label_index)

    n_classes = len(labels)
    for label_index in labels:
        # w_j= total_number_samples / (n_classes * n_samples_j)
        class_weights[label_index] = total_number_samples / (n_classes * occurrences[label_index])

    print('\tDone!')

    return dict(class_weights)


# Evaluates a NN model and returns all necessary evaluation metrics for our desired visualizations.
def full_evaluate_numpy(
        model: keras.Sequential,
        callbacks: list[keras.src.callbacks.Callback],
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        max_epochs: int = 30,
        batch_size: int = 32
) -> (Any, float, int, np.mat):
    # (History, Running Time in Seconds, Executed Epochs, Best Epoch, Confusion Matrix)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./models/checkpoints/{model.name}',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Calculate class weights
    class_weights = numpy_weights(y_train)

    # Train the model on the training set
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks + [model_checkpoint_callback],
        validation_data=(x_val, y_val),
        class_weight=class_weights
    )
    end_time = time.time()

    # Check how many epochs were effectively executed
    epochs = len(history.history['loss'])

    # Restore best weights
    model.load_weights(f'./models/checkpoints/{model.name}')

    # Evaluate Loss by applying the trained model to the test set
    metric_values = model.evaluate(x_test, y_test, batch_size=32)

    # Get the model's predictions on the test set and build a confusion matrix
    predicted = model.predict(x_test)

    y_true = y_test.argmax(axis=1)
    y_pred = predicted.argmax(axis=1)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    return history, metric_values, end_time - start_time, epochs, conf_matrix


def full_evaluate_tensor(
        model: keras.Sequential,
        callbacks: list[keras.src.callbacks.Callback],
        train_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict[int, float],
        max_epochs: int = 30,
        batch_size: int = 32
) -> (Any, float, int, np.mat):
    # (History, Running Time in Seconds, Executed Epochs, Best Epoch, Confusion Matrix)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./models/checkpoints/{model.name}',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Train the model on the training set
    start_time = time.time()
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks + [model_checkpoint_callback],
        validation_data=val_ds,
        class_weight=class_weights
    )
    end_time = time.time()

    # Check how many epochs were effectively executed
    epochs = len(history.history['loss'])

    # Restore best weights
    model.load_weights(f'./models/checkpoints/{model.name}')

    # Evaluate Loss by applying the trained model to the test set
    metric_values = model.evaluate(test_ds)

    # Get the model's predictions on the test set
    # https://stackoverflow.com/questions/64622210/how-to-extract-classes-from-prefetched-dataset-in-tensorflow-for-confusion-matri
    y_true = []
    y_pred = []
    for (x_test_batch, y_test_batch) in test_ds:
        y_true.append(y_test_batch)
        prediction = model.predict(x_test_batch)
        y_pred.append(np.argmax(prediction, axis= -1))

    # Convert the true and predicted labels into tensors
    correct_labels = [np.argmax(onehot) for onehot in tf.concat([item for item in y_true], axis=0)]
    predicted_labels = tf.concat([item for item in y_pred], axis=0)

    conf_matrix = confusion_matrix(y_true=correct_labels, y_pred=predicted_labels)

    return history, metric_values, end_time - start_time, epochs, conf_matrix
