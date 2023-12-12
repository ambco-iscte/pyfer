import numpy as np
import yaml
import cv2 as cv
import tensorflow as tf

from keras import Model


class EmotionClassifier:

    def __init__(self, classifier: Model, config_file_path: str):
        self.classifier = classifier
        self.emotions: dict[int, str] = yaml.load(open(config_file_path), yaml.CLoader)['emotions']

    def classify(self, image: np.ndarray) -> (str, float):
        # Get shape accepted by classification model
        n, w, h, c = self.classifier.input_shape

        # Resize image to that size
        img = cv.resize(image, (w, h))

        # Add an outer axis, i.e. (w,h,c) goes to (1,w,h,c) so model doesn't scream
        img = tf.expand_dims(img, axis=0)

        # Pass image to model and get prediction
        prediction = self.classifier(img).numpy()[0]

        # Convert softmax probabilities to class with the highest probability
        emotion_index = np.argmax(prediction)

        # Return emotion based on which probability was highest
        if emotion_index not in self.emotions.keys():
            return 'INVALID'
        return self.emotions[emotion_index], float(prediction[emotion_index])
