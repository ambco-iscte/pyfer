from ultralytics import YOLO
from ultralytics.engine.model import Model
from model.detector.bounds import BoundingBox


class FaceDetector:

    def __init__(self, model: Model):
        self.model = model

    def train(self, config: str, epochs: int = 100, patience: int = 30, output: str = "yolov8-face-detector"):
        return self.model.train(data=config, epochs=epochs, patience=patience)

    def detect(self, image: str) -> list[BoundingBox]:
        return BoundingBox.wrap(self.model(image)[0])


if __name__ == "__main__":
    yolo = FaceDetector(YOLO('yolov8n.pt'))
    res = yolo.train('config.yaml')
    print(res)
