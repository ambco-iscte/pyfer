import numpy as np

from model.detector.bounds import BoundingBox
from model.detector.detector import FaceDetector
from model.recogniser.classifier import EmotionClassifier


class PyFER:

    def __init__(self, detector: FaceDetector, classifier: EmotionClassifier):
        self.detector = detector
        self.classifier = classifier

    def apply(self, image: np.ndarray) -> list[BoundingBox]:
        results: list[BoundingBox] = []

        # Detect faces and get image fragments for bounding boxes
        detected = self.detector.detect(image)

        # Classify each face
        for box in detected:
            face = image[box.start[0]:box.start[1], box.end[0]:box.end[1]]
            box.name = self.classifier.classify(face)
            results.append(box)

        return results
