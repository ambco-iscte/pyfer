import numpy as np
import yaml

from keras import Model


class EmotionClassifier:

    def __init__(self, classifier: Model):
        self.classifier = classifier
        self.emotions: dict[int, str] = yaml.load('affectnet/config.yaml', yaml.CLoader)['emotions']

    def classify(self, image: np.ndarray) -> str:
        prediction = self.classifier(image)
        emotion = prediction.argmax(axis=1)
        if emotion not in self.emotions.keys():
            return 'INVALID'
        return self.emotions[emotion]
