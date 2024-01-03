import cv2 as cv
import tensorflow as tf
import torch
from keras.models import load_model
from ultralytics import YOLO

from model.classifier.classifier import EmotionClassifier
from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from preprocess import affectnet

DETECTOR = 'trained-models/detector.pt'
CLASSIFIER = 'model/classifier/training/models/AffectNetFromScratch.keras'
CLASSIFIER_CONFIG = 'model/classifier/training/affectnet/config.yaml'


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
    video = cv.VideoCapture(0)

    # Set video capture properties
    # Necessary because we're running Python on WSL2, which makes it a major headache to access a USB camera.
    # If you're running Python natively, try removing these options!
    video.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    video.set(cv.CAP_PROP_FPS, 15)
    video.set(cv.CAP_PROP_FOURCC,  cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    if not video.isOpened():
        raise RuntimeError("Failed to open video capture.")

    while(True):
        if not video.isOpened():
            break

        ret, frame = video.read()

        try:
            frame = affectnet(frame)

            detections = pyfer.apply(frame)
            frame_processed = annotated(frame, detections)

            cv.imshow('PyFER Webcam Capture', cv.cvtColor(frame_processed, cv.COLOR_RGB2BGR))
        except:
            continue

        # Quit when user presses the Q key
        if cv.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
