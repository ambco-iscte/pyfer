from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import mobilenet_v3_small

from model.recogniser.dataset import load

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report


# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
# https://machinelearningmastery.com/training-and-validation-data-in-pytorch/


def fit(
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
        x_train,
        x_val,
        y_train,
        y_val,
        epochs: int = 100,
        batch_size: int = 10
) -> dict[str, list[float]]:
    """
    Fits a Torch NN model to a given set of data and calculates evaluation metrics along the training process.
    :param model: The Torch NN model to be fitted.
    :param loss_function: Loss function to use.
    :param optimizer: Gradient optimizer to use.
    :param x_train: TODO
    :param x_val: TODO
    :param y_train: TODO
    :param y_val: TODO
    :param epochs: Number of epochs to run.
    :param batch_size: TODO
    :return: A dictionary containing the history of training and validation set metrics along the training process.
    """
    metrics: dict[str, list[float]] = defaultdict()

    def append(name: str, x_true, y_true):
        predicted = model(x_true)
        metrics[f'{name}_loss'].append(loss_function(predicted, y_true))
        metrics[f'{name}_accuracy'].append(accuracy_score(predicted, y_true))
        metrics[f'{name}_precision'].append(precision_score(predicted, y_true))
        metrics[f'{name}_recall'].append(recall_score(predicted, y_true))
        metrics[f'{name}_f1'].append(f1_score(predicted, y_true))

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_pred = model(x_batch)
            y_batch = y_train[i:i + batch_size]
            loss = loss_function(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    append('train', x_train, y_train)
    append('val', x_val, y_val)

    return metrics


def main():
    # Build model
    # TODO:
    #  How do we set trainable=False?
    #  How do we set include_top?
    pretrained_model = mobilenet_v3_small(pretrained=True)
    model = nn.Sequential(
        pretrained_model,  # TODO: Remember to rescale images by 1./255

        # Flatten data to be able to use dense layers
        nn.Flatten(),

        # Dense layer with ReLU activation
        nn.Linear(),  # TODO: figure out in_features and out_features
        nn.ReLU(),

        # Dropout to avoid over-fitting
        nn.Dropout(0.2),

        # Dense layer with Softmax activation
        nn.Linear(),  # TODO: figure out in_features and out_features (is out_features = num_classes here?)
        nn.Softmax()
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Get data
    x_train, x_val, x_test, y_train, y_val, y_test = load(
        'fer2013.csv',
        'config.yaml',
        0.8,
        0.1
    )

    # Fit model to training data
    epochs = 100
    batch_size = 10
    history = fit(model, loss_function, optimizer, x_train, x_val, y_train, y_val, epochs, batch_size)
    # TODO: plot history

    with torch.no_grad():
        y_pred = model(x_test)
        report = classification_report(y_test, y_pred)
        # TODO: analyse report
        
    # TODO: save model to file if wanted


if __name__ == '__main__':
    main()
