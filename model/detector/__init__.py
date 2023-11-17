from ultralytics import YOLO

from model.detector.files import load_images

FACES_FILE = "detector_training_data/faces.csv"
IMAGE_FOLDER = "detector_training_data/images"
#images = load_images(IMAGE_FOLDER, FACES_FILE)

# Load a model
# The letter at the end of the model name indicates size: n < s < m < l < x
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Train the model
    results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

    # Use the model
    model.train(data="custom_data.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format

