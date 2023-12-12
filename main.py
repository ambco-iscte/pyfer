import time

import torch
import cv2 as cv
import numpy as np

from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.recogniser.classifier import EmotionClassifier
from keras.models import load_model

DETECTOR = 'trained-models/detector.pt'
CLASSIFIER = 'trained-models/classifier'


def main():
    # Set PyTorch to use GPU (big speedup for YOLO if CUDA is installed)
    torch.cuda.set_device(0)

    # Load detector and classifier models
    detector = FaceDetector(YOLO(DETECTOR))
    classifier = EmotionClassifier(load_model(CLASSIFIER), 'model/recogniser/fer/config.yaml')

    # Instantiate PyFER model
    pyfer = PyFER(detector, classifier)

    # Load image
    image: np.ndarray = cv.imread('testing.jpg')
    # cv.imshow('Original Image', image)

    # Detect and classify faces
    print(f'Applying PyFER to image of shape {image.shape}...')
    start_time = time.time()
    detections = pyfer.apply(image)
    end_time = time.time()
    print(f'Done! Took {end_time - start_time} seconds.')

    # Show image with detected faces and emotions
    image_processed = annotated(image, detections)
    cv.imshow('PyFER Image', image_processed)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
