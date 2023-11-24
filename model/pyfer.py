import torch.nn as nn

from model.detector.detector import FaceDetector
from model.recogniser.classifier import EmotionClassifier


class PyFer:

    def __init__(self, detector: FaceDetector, classifier: EmotionClassifier):
        self.detector = detector
        self.classifier = classifier
