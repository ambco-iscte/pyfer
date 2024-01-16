from enum import Enum

import keras
from keras import layers, metrics, losses, optimizers
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.applications.resnet_v2 import ResNet50V2


class TransferLearningType(Enum):
    NONE = 0,
    MOBILENET = 1,
    RESNET = 2


def build(
        name: str,
        transfer_learning_type: TransferLearningType,
        input_shape: (int, int),
        finetuning: int,
        num_classes: int,
        learning_rate: float
) -> keras.Model:
    """
    Builds a Keras Neural Network model from the given specification.
    :param name: Model name.
    :param transfer_learning_type: What type of transfer learning should be used.
    :param input_shape: Input image shape.
    :param finetuning: Amount of pre-trained model layers to unlock, if applicable.
    :param num_classes: Number of categorical output classes.
    :param learning_rate: Learning rate for the model.
    """
    assert 0 < learning_rate <= 1

    img_width, img_height = input_shape

    if transfer_learning_type == TransferLearningType.MOBILENET:
        convolutional_model = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False)
        convolutional_model.trainable = False
        for layer in convolutional_model.layers[-finetuning:]:
            layer.trainable = True
    elif transfer_learning_type == TransferLearningType.RESNET:
        convolutional_model = ResNet50V2(input_shape=(img_width, img_height, 3), include_top=False)
        convolutional_model.trainable = False
        for layer in convolutional_model.layers[-finetuning:]:
            layer.trainable = True
    else:
        convolutional_model = keras.Sequential([
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

    # CNN using categorical classification with softmax activation
    model = keras.Sequential([
        # Data Augmentation
        layers.RandomFlip("horizontal", input_shape=(img_width, img_height, 3)),
        layers.RandomRotation(0.2),
        
        layers.Rescaling(2./255, offset=-1),  # Rescale to [-1, 1]

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
    ], name=name)

    model.compile(
        loss=losses.CategoricalCrossentropy(),  # Calculates entropy for multi-class classification problems
        optimizer=optimizers.AdamW(learning_rate=learning_rate),  # Variation of Gradient Descent

        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            # metrics.F1Score(name='f1_score')
        ]
    )

    return model
