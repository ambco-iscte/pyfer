import cv2 as cv
import numpy as np

from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.recogniser.classifier import EmotionClassifier
from keras.models import load_model

DETECTOR = 'model/detector/yolov8n.pt'
CLASSIFIER = ''


def main():
    # Load detector and classifier models
    detector = FaceDetector(YOLO(DETECTOR))
    classifier = EmotionClassifier(load_model(CLASSIFIER))

    # Instantiate PyFER model
    pyfer = PyFER(detector, classifier)

    # Load image
    image: np.ndarray = cv.imread('testing.png')
    cv.imshow('Original Image', image)

    # Detect and classify faces
    detections = pyfer.apply(image)
    image_processed = annotated(image, detections)
    cv.imshow('PyFER Image', image_processed)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
