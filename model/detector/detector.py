import numpy as np
import cv2 as cv

from ultralytics import YOLO
from ultralytics.engine.model import Model
from model.detector.bounds import BoundingBox, annotated


class FaceDetector:
    """
    Wrapper class for an Ultralytics object detection model.
    """

    def __init__(self, model: Model):
        self.model = model

    def train(self, config: str, epochs: int = 100, patience: int = 30):
        return self.model.train(data=config, epochs=epochs, patience=patience)

    # https://docs.ultralytics.com/modes/predict/#inference-sources
    # HWC format with BGR channels uint8 (0-255).
    def detect(self, image: str | np.ndarray) -> list[BoundingBox]:
        return BoundingBox.wrap(self.model(image, verbose=False)[0])


if __name__ == "__main__":
    yolo = FaceDetector(YOLO('../../trained-models/detector.pt'))
    # res = yolo.train('config.yaml')
    # print(res)

    img = cv.imread('testing.jpg')
    boxes = yolo.detect(img)
    cv.imshow('Detections', annotated(img, boxes))

    cv.waitKey(0)
    cv.destroyAllWindows()
