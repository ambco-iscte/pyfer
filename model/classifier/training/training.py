import datetime
import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from keras.src.callbacks import EarlyStopping

from model.classifier.training.affectnet.dataset import load_affectnet
from model.classifier.training.builder import TransferLearningType
from model.classifier.training.builder import build
from model.classifier.training.fer.dataset import load as load_fer
from model.classifier.training.tensorutils import plot_metric, plot_confusion_matrix, full_evaluate_numpy, \
    full_evaluate_tensor, dataset_weights


def plot(model_name, labels, history, metric_values, time_value, epochs, conf_matrix):
    time_value = str(datetime.timedelta(seconds=round(time_value)))

    save_folder = os.path.join('results', model_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    def make_metric_plot(metric: str):
        f, a = plt.subplots()
        plot_metric(
            a,
            metric,
            history,
            f'{model_name}\nLoss\nProgression over {epochs} epochs\n\nRun Time = {time_value}s\n',
            'Epoch',
            metric.capitalize()
        )
        f.canvas.manager.set_window_title(
            f'PyFER - Progression of {metric.capitalize()} over Training for {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'{metric}.png'), bbox_inches="tight")
        plt.close(f)

    # Plot Metrics
    make_metric_plot('accuracy')
    make_metric_plot('loss')
    make_metric_plot('precision')
    make_metric_plot('recall')

    # Confusion Matrix
    fig, ax = plt.subplots()
    cfm_display = plot_confusion_matrix(
        conf_matrix,
        ax,
        f"{model_name}\n\n{time_value} over {epochs} epochs",
        labels,
        ['loss', 'accuracy', 'precision', 'recall'],
        metric_values
    )
    fig.canvas.manager.set_window_title(f'PyFER - Confusion Matrix for {model_name}')
    plt.tight_layout()
    cfm_display.figure_.savefig(os.path.join(save_folder, f'confusion_matrix.png'), bbox_inches="tight")
    plt.close(cfm_display.figure_)


def execute(
        input: Sequence,
        labels: list[str],
        class_weights: dict[int, float] | None,
        transfer_learning_type: TransferLearningType,
        input_shape: (int, int),
        finetuning: int,
        patience: int,
        model_name: str,
        learning_rate: float,
        max_epochs: int,
        batch_size: int
):
    """

    :param input: Sequence of input sets. Should be (x_train, y_train, x_val, y_val, x_test, y_test) for numpy arrays,
        or (train_ds, val_ds, test_ds) for Keras Datasets.
    :param class_weights: Dictionary mapping each class index to a weight.
    :param labels: A list of class names.
    :param transfer_learning_type: What type of transfer learning should be used.
    :param input_shape: Desired image input shape.
    :param finetuning: Amount of pre-trained model layers to unlock, if applicable.
    :param patience: Stop training early if validation loss has stagnated for this number of epochs.
    :param model_name: Model name.
    :param learning_rate: Learning rate for the model.
    :param max_epochs: Maximum number of allowed training epochs.
    :param batch_size: Training batch size.
    :return:
    """
    num_classes = len(labels)
    print(f"\nUsing {num_classes} classes:\n\t" + '\n\t'.join(labels))

    img_width, img_height = input_shape

    # Build Model
    model = build(
        model_name,
        transfer_learning_type,
        (img_width, img_height),
        finetuning,
        num_classes,
        learning_rate
    )
    model.summary()
    print()

    # Is the input sequence formatted to use numpy arrays?
    is_numpy_input = class_weights is None and len(input) == 6 and all(isinstance(s, np.ndarray) for s in input)

    # Is the input sequence formatted to use Keras datasets?
    is_keras_input = class_weights is not None and len(input) == 3 and all(isinstance(s, tf.data.Dataset) for s in input)

    # Stop training early if the loss of the validation set has stagnated.
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    # Evaluate the Model
    if is_numpy_input:
        x_train, y_train, x_val, y_val, x_test, y_test = input
        history, metric_values, time_value, epochs, conf_matrix = full_evaluate_numpy(
            model,
            [early_stopping],
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            max_epochs=max_epochs,
            batch_size=batch_size
        )
    elif is_keras_input:
        train_ds, val_ds, test_ds = input
        history, metric_values, time_value, epochs, conf_matrix = full_evaluate_tensor(
            model,
            [early_stopping],
            train_ds,
            test_ds,
            val_ds,
            class_weights,
            max_epochs=max_epochs,
            batch_size=batch_size
        )
    else:
        raise AssertionError('Model input did not match required format. Should be either '
                             '(x_train, y_train, x_val, y_val, x_test, y_test) or (train_ds, val_ds, test_ds).')
    model.save(f'./models/{model_name}')

    # Plot Model Results
    plot(model_name, labels, history, metric_values, time_value, epochs, conf_matrix)


def fer(transfer_learning_type: TransferLearningType, model_name: str, model_config) -> (int, int, int):
    # Get data
    x_train, x_val, x_test, y_train, y_val, y_test = load_fer(
        'fer/fer2013.csv',
        'fer/fer2013new.csv',
        'fer/config.yaml',
        0.8,
        0.1
    )
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Get configuration
    config = yaml.load(open('fer/config.yaml'), yaml.CLoader)
    emotions = config['emotions']

    # Get image width and height
    print(x_train[0].shape)
    img_width, img_height, _ = x_train[0].shape

    # Execute model
    execute(
        input=(x_train, y_train, x_val, y_val, x_test, y_test),
        labels=emotions.values(),
        class_weights=None,
        transfer_learning_type=transfer_learning_type,
        input_shape=(img_width, img_height),
        finetuning=model_config['FER']['finetuning'],
        patience=model_config['FER']['patience'],
        model_name=model_name,
        learning_rate=model_config['FER']['learning_rate'],
        max_epochs=model_config['FER']['max_epochs'],
        batch_size=model_config['FER']['batch_size']
    )

    return len(x_train), len(x_val), len(x_test)


def affectnet(
        transfer_learning_type: TransferLearningType,
        name: str,
        model_config,
        sample_sizes: (int, int, int) = (-1, -1, -1),
        shape: (int, int) = (224, 224)
):
    # Get data
    train_ds, test_ds, val_ds = load_affectnet(
        'affectnet/config.yaml',
        'affectnet/data',
        shape=shape
    )

    # Sample training set (too big of a set causes excessively long training times and/or memory exceptions)
    sample_size_train, sample_size_val, sample_size_test = sample_sizes
    if sample_size_train > 0:
        train_ds = train_ds.take(sample_size_train)
    if sample_size_val > 0:
        val_ds = val_ds.take(sample_size_val)
    if sample_size_test > 0:
        test_ds = test_ds.take(sample_size_test)

    # Get configuration
    config = yaml.load(open('affectnet/config.yaml'), yaml.CLoader)
    emotions = config['emotions']

    # Execute model
    execute(
        input=(train_ds, val_ds, test_ds),
        labels=emotions.values(),
        class_weights=dataset_weights(train_ds),
        transfer_learning_type=transfer_learning_type,
        input_shape=shape,
        finetuning=model_config['AffectNet']['finetuning'],
        patience=model_config['AffectNet']['patience'],
        model_name=name,
        learning_rate=model_config['AffectNet']['learning_rate'],
        max_epochs=model_config['AffectNet']['max_epochs'],
        batch_size=model_config['AffectNet']['batch_size']
    )


if __name__ == "__main__":
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f'Using GPU for Keras model training! Yay!')
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    # Disable TensorFlow warnings
    logging.disable(logging.WARNING)
    logging.disable(logging.INFO)
    tf.get_logger().setLevel('INFO')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Get universal configuration
    cfg = yaml.load(open('config.yaml'), yaml.CLoader)

    # FER+
    fer(TransferLearningType.MOBILENET, 'FERPlusUsingMobileNet', cfg)
    fer(TransferLearningType.RESNET, 'FERPlusUsingResNet', cfg)
    fer(TransferLearningType.NONE, 'FERPlusFromScratch', cfg)

    # AffectNet
    sample_sizes = (50000, -1, -1)
    affectnet(TransferLearningType.MOBILENET, 'AffectNetUsingMobileNet', cfg, shape=(128, 128), sample_sizes=sample_sizes)
    affectnet(TransferLearningType.RESNET, 'AffectNetUsingResNet', cfg, shape=(128, 128), sample_sizes=sample_sizes)
    affectnet(TransferLearningType.NONE, 'AffectNetFromScratch', cfg, shape=(128, 128), sample_sizes=sample_sizes)
