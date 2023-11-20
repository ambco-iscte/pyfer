from ultralytics import YOLO

from model.detector.files import create_labels, shuffle_and_split_files

FACES_FILE = "detector_training_data/faces.csv"
IMAGE_FOLDER = "detector_training_data/images"
#images = load_images(IMAGE_FOLDER, FACES_FILE)

#create_labels(IMAGE_FOLDER, FACES_FILE, "detector_training_data/labels")

shuffle_and_split_files(IMAGE_FOLDER)

# Load a model
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

#!yolo task=detect mode=train model=yolov8x.pt data=/content/data.yaml epochs=100 imgsz=640 batch=8 project=/content/drive/report  "save=True"

#Project = is a destination where u want to save some valuable results matrix

# Train the model
#results = model.train(data='coco128.yaml', epochs=100, imgsz=640)