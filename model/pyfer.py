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
            face = image[box.start[1]:box.end[1], box.start[0]:box.end[0]]  # [y1:y2, x1:x2]
            emotion, confidence = self.classifier.classify(face)
            box.name = f'{emotion} ({round(confidence, 3)})'
            results.append(box)

        return results
