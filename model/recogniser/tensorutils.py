import time
import keras.src.callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from typing import Any

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


def decode(item: list[int], mapping: dict[int, str]) -> str:
    """
    Decodes a one-hot-encoded element according to a given dictionary mapping each index to a class.
    :param item: One-hot-encoded element.
    :param mapping: Dictionary mapping each index to a class.
    :return:
    """
    return mapping[np.argmax(item)]


# Displays a (width * height) grid of plots, each showcasing one element from the provided data.
# Useful since we're using images and this allows us to visualize a bunch of them in a single, neat plot. :)
def visualize(width: int, height: int, x, y, decoding: dict[int, str]):
    fig, ax = plt.subplots(height, width)
    for i in range(height):
        for j in range(width):
            ax[i, j].imshow(x[i * height + j], cmap=plt.get_cmap('gray'))
            ax[i, j].set_title(decoding[np.argmax(y[i * height + j])])
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.canvas.manager.set_window_title('PyFER - Input Images')
    plt.suptitle(f'First {width * height} Training Samples')
    plt.tight_layout()


# Generates a plot for a trained model's metric.
# Useful for displaying accuracy, loss, etc.
def plot_metric(axis, metric, history, title, x_label, y_label, best_epoch):
    axis.plot(history.history[metric])
    axis.plot(history.history[f'val_{metric}'])
    axis.axvline(best_epoch, linestyle='--', color='g')
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)
    axis.legend(['Training', 'Validation', 'Best Validation Accuracy'], loc="upper left")


def plot_confusion_matrix(matrix, axis, title, labels, metrics, metric_values):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=axis, colorbar=False, cmap="Blues")
    cf_title = f'{title}\n'
    for i, metric in enumerate(metrics):
        cf_title += f"\n{metric.capitalize()} = {round(metric_values[i], 2)}"
    axis.set_title(cf_title)


# Evaluates a NN model and returns all necessary evaluation metrics for our desired visualizations.
def full_evaluate(
        model: tf.keras.Sequential,
        callbacks: list[keras.src.callbacks.Callback],
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        categorical: bool,
        postprocessor=None,
        max_epochs: int = 30,
        batch_size: int = 32
) -> (Any, float, int, int, np.mat):
    # (History, Running Time in Seconds, Executed Epochs, Best Epoch, Confusion Matrix)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./models/{model.name}',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Train the model on the training set
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        verbose=1,
        callbacks=callbacks + [model_checkpoint_callback],
        validation_data=(x_val, y_val)
    )
    end_time = time.time()

    # Check how many epochs were effectively executed
    epochs = len(history.history['loss'])

    # Check which epoch was the best
    best_epoch = np.argmax(history.history['val_accuracy'])

    # Restore best weights
    model.load_weights(f'./models/{model.name}')

    # Evaluate Loss by applying the trained model to the test set
    metric_values = model.evaluate(x_test, y_test, batch_size=32)

    # Get the model's predictions on the test set and build a confusion matrix
    predicted = model.predict(x_test)
    y_true, y_pred = y_test, predicted

    if postprocessor is not None:
        y_pred = postprocessor(y_pred)

    if categorical:
        y_true = y_test.argmax(axis=1)  # Convert to list of actual classes, which are just indices, because digits :)
        y_pred = predicted.argmax(axis=1)  # Convert to list of predicted classes
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    return history, metric_values, end_time - start_time, epochs, best_epoch, conf_matrix
