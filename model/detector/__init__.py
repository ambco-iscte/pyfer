from model.detector.files import load_images

FACES_FILE = "detector_training_data/faces.csv"
IMAGE_FOLDER = "detector_training_data/images"
images = load_images(IMAGE_FOLDER, FACES_FILE)
