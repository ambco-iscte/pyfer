import cv2 as cv

from ultralytics import YOLO
from model.detector.bounds import BoundingBox, annotated


class FaceDetector:

    def __init__(self, model: str):
        self.model = YOLO(model)

    def train(self, config: str, epochs: int = 100, patience: int = 30, output: str = "yolov8-face-detector"):
        return self.model.train(data=config, epochs=epochs, patience=patience)

    def detect(self, image: str) -> list[BoundingBox]:
        return BoundingBox.wrap(self.model(image)[0])


if __name__ == "__main__":
    # yolo = FaceDetector('yolov8n.pt')
    # res = yolo.train('config.yaml')
    # print(res)

    yolo = FaceDetector('./runs/detect/train/weights/best.pt')
    boxes = yolo.detect("photo.jpg")
    for box in boxes:
        print(box)

    img = cv.imread("photo.jpg")
    cv.imshow("Image", annotated(img, boxes))

    cv.waitKey(0)
    cv.destroyAllWindows()
