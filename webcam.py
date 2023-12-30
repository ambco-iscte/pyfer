import cv2 as cv
import tensorflow as tf
import torch
from keras.models import load_model
from ultralytics import YOLO

from model.classifier.classifier import EmotionClassifier
from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER

DETECTOR = 'trained-models/detector.pt'
CLASSIFIER = 'trained-models/fer/FERPlusFromScratch.keras'
CLASSIFIER_CONFIG = 'trained-models/fer/config.yaml'


def main():
    # Set PyTorch to use GPU (big speedup for YOLO if CUDA is installed)
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f'Using GPU for Keras models! Yay!')
    torch.cuda.set_device(0)

    print(f'Loading detector and classifier models...')

    # Load detector and classifier models
    detector = FaceDetector(YOLO(DETECTOR))
    classifier = EmotionClassifier(load_model(CLASSIFIER), CLASSIFIER_CONFIG)

    # Instantiate PyFER model
    pyfer = PyFER(detector, classifier)
    print('Done! Starting video capture...')

    # Start video capture
    video = cv.VideoCapture(-1)

    while(True):
        if not video.isOpened():
            break

        ret, frame = video.read()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        detections = pyfer.apply(frame)
        frame_processed = annotated(frame, detections)

        cv.imshow('PyFER Webcam Capture', cv.cvtColor(frame_processed, cv.COLOR_RGB2BGR))

        # Quit when user presses the Q key
        if cv.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
