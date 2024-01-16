import time

import cv2 as cv
import numpy as np
import torch
from keras.models import load_model
from ultralytics import YOLO

from model.classifier.classifier import EmotionClassifier
from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER

DETECTOR = 'trained-models/detector.pt'
CLASSIFIER = 'path/to/classifier'                       # Change as needed
CLASSIFIER_CONFIG = 'path/to/classifier/config.yaml'    # Change as needed


def main():
    torch.cuda.set_device(0)

    # Load detector and classifier models
    detector = FaceDetector(YOLO(DETECTOR))
    classifier = EmotionClassifier(load_model(CLASSIFIER), CLASSIFIER_CONFIG)

    # Instantiate PyFER model
    pyfer = PyFER(detector, classifier)

    # Load image
    image: np.ndarray = cv.cvtColor(
        cv.imread('path/to/image.png'),  # Change as needed
        cv.COLOR_BGR2RGB
    )

    # Detect and classify faces
    print(f'Applying PyFER to image of shape {image.shape}...')
    start_time = time.time()
    detections = pyfer.apply(image)
    end_time = time.time()
    print(f'Done! Took {end_time - start_time} seconds.')

    # Show image with detected faces and emotions
    image_processed = annotated(image, detections)
    cv.imshow('PyFER Image', cv.cvtColor(image_processed, cv.COLOR_RGB2BGR))

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
