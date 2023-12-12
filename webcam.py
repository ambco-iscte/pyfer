import cv2 as cv
import torch
from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.recogniser.classifier import EmotionClassifier
from keras.models import load_model

DETECTOR = 'trained-models/detector.pt'
CLASSIFIER = 'trained-models/classifier'
CLASSIFIER_CONFIG = 'trained_models/classifier_config.yaml'


def main():
    # Set PyTorch to use GPU (big speedup for YOLO if CUDA is installed)
    torch.cuda.set_device(0)

    print(f'Loading detector and classifier models...')

    # Load detector and classifier models
    detector = FaceDetector(YOLO(DETECTOR))
    classifier = EmotionClassifier(load_model(CLASSIFIER), CLASSIFIER_CONFIG)

    # Instantiate PyFER model
    pyfer = PyFER(detector, classifier)
    print('Done! Starting video capture...')

    # Start video capture
    video = cv.VideoCapture(0)

    while(True):
        if not video.isOpened():
            break

        ret, frame = video.read()

        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        detections = pyfer.apply(frame)
        frame_processed = annotated(frame, detections)

        cv.imshow('PyFER Webcam Capture', frame_processed)

        # Quit when user presses the Q key
        if cv.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
