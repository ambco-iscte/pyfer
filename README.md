<div align="center">

# PyFER
**Automated Facial Expression Emotion Recognition Using Chained Neural Networks in Python**

[![PDF - Check out our project report!](https://img.shields.io/badge/PDF-Check_out_our_project_report!-3172c8?logo=Adobe)](report.pdf)



</div>

## Context
This library was developed as the final project for an elective
[Deep Learning for Computer Vision](https://fenix-mais.iscte-iul.pt/courses/04193-565977905365459) 
course as part of the 
[Master's (MSc) in Computer Engineering](https://www.iscte-iul.pt/course/12/master-msc-in-computer-engineering) 
programme at [Iscte-IUL](https://www.iscte-iul.pt/).

<br>

## How to Use
**PyFER** relies on two separate neural network models to automatically detect and classify
facial expressions in any given image:
1. A [FaceDetector](model/detector/detector.py) model;
2. An [EmotionClassifier](model/recogniser/classifier.py) model.

A **FaceDetector** object requires an [Ultralytics](https://github.com/ultralytics/ultralytics) model
(these are built on PyTorch) to be created.

An **EmotionClassifier** object, in turn, requires:
1. A [TensorFlow](https://www.tensorflow.org/) model;
2. The path to a `yaml` configuration file containing an `emotions` mapping between integers and emotion names.

Ideally, EmotionClassifier would also support PyTorch models to facilitate faster native execution on GPUs.
However, we ran into a headache-inducing amount of issues when trying to support PyTorch models and, since this library
was developed for a single university assignment, we decided to stick to the familiarity of Keras instead. :)

Specifically, a PyFER model could be instantiated as follows.
```python
# Load detector and classifier models
detector = FaceDetector(ultralytics_model)
classifier = EmotionClassifier(
    classifier=tensorflow_model,
    config_file_path='path/to/classifier/config/file.yaml'
)

# Instantiate PyFER model
pyfer = PyFER(detector, classifier)
```

And the following is an example of an EmotionClassifier configuration file.
```yaml
emotions:
  0: 'Neutral'
  1: 'Happiness'
  2: 'Surprise'
  3: 'Sadness'
  4: 'Anger'
  5: 'Disgust'
  6: 'Fear'
  7: 'Contempt'
  8: 'Unknown'
  9: 'NF'
```

### Using Pre-Trained Models
If you download this repository, you'll find two ready-to-use pre-trained models, one face detector and one
facial expression classifier. These are stored in the [trained-models](trained-models) folder.

These models can be readily used by simply specifying their paths when instantiating the PyFER models.
```python
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = EmotionClassifier(
    classifier=load_model('trained-models/classifier'),
    config_file_path='trained_models/classifier_config.yaml'
)

pyfer = PyFER(detector, classifier)
```

### Creating your own Models
**Object Detection**: 
- Please refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes). :)
- Make sure to [train](https://docs.ultralytics.com/modes/train/) your model to only detect faces!
- Check out the [detector.py](model/detector/detector.py) file to see how we did this.

**Facial Expression Classification**:
- Construct any TensorFlow/Keras model that receives an image as input and, using Softmax or any similar activation at the output layer, outputs the probabilities of that image belonging to each given facial expression emotion class;
- Assign an integer value (preferably 0-N) to each of your considered emotions, and one-hot encode target labels using that mapping.
- Check out the [recogniser.py](model/recogniser/recogniser.py) file to see how we did this.

<br>

## Example
### Applying PyFER to a single image
The following is an example of applying PyFER to a single image. Try running [main.py](main.py) on your machine!
```python
import cv2 as cv

from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.recogniser.classifier import EmotionClassifier
from keras.models import load_model


# Load detector and classifier models
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = EmotionClassifier(
    classifier=load_model('trained-models/classifier'),
    config_file_path='trained_models/classifier_config.yaml'
)

# Instantiate PyFER model
pyfer = PyFER(detector, classifier)

# Load image
image = cv.imread('path/to/image.png')
cv.imshow('Original Image', image)

# Detect and classify faces
detections = pyfer.apply(image)
image_processed = annotated(image, detections)
cv.imshow('PyFER Image', image_processed)

cv.waitKey(0)
cv.destroyAllWindows()
```

### Applying PyFER to webcam feed
If the models making up PyFER can be executed on the GPU and their execution is fast enough,
PyFER can be applied to the frames of a webcam feed to automatically detect and classify the emotions
of people in that feed in close to real-time!

The following is an example of this. Try running [webcam.py](webcam.py) on your machine!
```python
import cv2 as cv
import torch
from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.recogniser.classifier import EmotionClassifier
from keras.models import load_model


# Set PyTorch to use GPU (big speedup for YOLO if CUDA is installed)
torch.cuda.set_device(0)

# Load detector and classifier models
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = EmotionClassifier(
    classifier=load_model('trained-models/classifier'),
    config_file_path='trained_models/classifier_config.yaml'
)

# Instantiate PyFER model
pyfer = PyFER(detector, classifier)

# Start video capture
video = cv.VideoCapture(0)  # Might need to adjust this number

while(True):
    if not video.isOpened():
        break

    # Read frame from webcam video capture
    ret, frame = video.read()

    # Apply PyFER to this frame
    detections = pyfer.apply(frame)
    frame_processed = annotated(frame, detections)

    # Display the annotated frame
    cv.imshow('PyFER Webcam Capture', frame_processed)

    # Quit when user presses the Q key
    if cv.waitKey(1) and 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
```


<br>

## Acknowledgements
We kindly thank Dr. Mohammad H. Mahoor, Professor of Electrical and Computer Engineering at the University of Denver, 
and M. Mehdi Hosseini, Ph.D. Student of Electrical and Computer Engineering at the University of Denver, for providing 
us with the [AffectNet dataset](http://mohammadmahoor.com/affectnet/) to aid in the development of our facial 
expression classification model.

We kindly thank Dr. Jeffrey Cohn and Megan Ritter from the University of Pittsburgh for providing us with the 
[Cohn-Kanade dataset](https://ieeexplore.ieee.org/document/5543262) and its extended version to aid in the development 
of our facial expression classification model. While we ended up not utilizing this dataset
to train our model, we appreciate being provided with it!

<br>

## Credit
Credit for all the code present in this repository goes to 
[Afonso Caniço](https://ciencia.iscte-iul.pt/authors/afonso-canico/cv)
and [Samuel Correia](https://www.linkedin.com/in/samuel0correia), 
authors and sole contributors to the project and this repository, 
unless otherwise explicitly stated.