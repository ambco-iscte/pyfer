<br>

<div align="center">

<img align="left" src="resources/PoweredByTensorFlow.png" alt="" style="height:3rem"/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/Ultralytics_YOLO_full_white.png">
  <source media="(prefers-color-scheme: light)" srcset="resources/Ultralytics_YOLO_full_blue.png">
  <img alt="Ultralytics" src="resources/Ultralytics_YOLO_full_blue.png" style="height:3rem; margin-left: 1.5rem">
</picture>

</div>

<br>

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
2. An [EmotionClassifier](model/classifier/classifier.py) model.

A **FaceDetector** object requires an [Ultralytics](https://github.com/ultralytics/ultralytics) model
(these are built on PyTorch) to be created.

An **EmotionClassifier** object, in turn, requires:
1. A [TensorFlow](https://www.tensorflow.org/) model; (Would, ideally, be PyTorch; TensorFlow is easier.)
2. The path to a `yaml` configuration file containing an `emotions` mapping between integers and emotion names.

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
```

### Using the Pre-Trained Models
If you download this repository, you'll find a ready-to-use YOLOv8-based face detection model stored in the [trained-models](trained-models) 
folder. It can be used by simply instantiating `FaceDetector` with it as an argument.
```python
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = ...

pyfer = PyFER(detector, classifier)
```

In the same folder you can additionally find 6 (six) pre-trained facial expression emotion classification models,
along with configuration files for the AffectNet and FER models.
```python
from keras.models import load_model

detector = ...
classifier = classifier = EmotionClassifier(load_model("trained-models/..."), "trained-models/config.yaml")

pyfer = PyFER(detector, classifier)
```
Try using each and see which one works best for you!



### Creating your own Models
**Object Detection**: 
- Please refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes). :)
- Make sure to [train](https://docs.ultralytics.com/modes/train/) your model to only detect faces!
- Check out the [detector.py](model/detector/detector.py) file to see how we did this.

**Facial Expression Classification**:
- Construct any TensorFlow/Keras model that receives an image as input and, using Softmax or any similar activation at the output layer, outputs the probabilities of that image belonging to each given facial expression emotion class;
- Assign an integer value to each of your considered emotions, and one-hot encode target labels using that mapping.
- Check out the [training.py](model/classifier/training/training.py) file to see how we did this.

<br>

## Example
### Applying PyFER to a single image
The following is an example of applying PyFER to a single image.

```python
import cv2 as cv

from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.classifier.classifier import EmotionClassifier
from keras.models import load_model

# Load detector and classifier models
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = EmotionClassifier(
    classifier=load_model('path/to/model'),
    config_file_path='path/to/yaml/config'
)

# Instantiate PyFER model
pyfer = PyFER(detector, classifier)

# Load image and convert to RGB
image = cv.cvtColor(cv.imread('path/to/image.png'), cv.COLOR_BGR2RGB)

# Detect and classify faces
detections = pyfer.apply(image)
image_processed = annotated(image, detections)
cv.imshow('PyFER Image', cv.cvtColor(image_processed, cv.COLOR_RGB2BGR))

cv.waitKey(0)
cv.destroyAllWindows()
```

### Applying PyFER to webcam feed
If the models making up PyFER can be executed on the GPU and their execution is fast enough,
PyFER can be applied to the frames of a webcam feed to automatically detect and classify the emotions
of people in that feed in close to real-time!

The following is an example of this.

```python
import cv2 as cv
import torch
from ultralytics import YOLO

from model.detector.bounds import annotated
from model.detector.detector import FaceDetector
from model.pyfer import PyFER
from model.classifier.classifier import EmotionClassifier
from keras.models import load_model

# Set PyTorch to use GPU (big speedup for YOLO if CUDA is installed)
torch.cuda.set_device(0)

# Load detector and classifier models
detector = FaceDetector(YOLO('trained-models/detector.pt'))
classifier = EmotionClassifier(
    classifier=load_model('path/to/model'),
    config_file_path='path/to/yaml/config'
)

# Instantiate PyFER model
pyfer = PyFER(detector, classifier)

# Start video capture
video = cv.VideoCapture(0)  # Might need to adjust this number

while (True):
    if not video.isOpened():
        break

    # Read frame from webcam video capture
    ret, frame = video.read()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Apply PyFER to this frame
    detections = pyfer.apply(frame)
    frame_processed = annotated(frame, detections)

    # Display the annotated frame
    cv.imshow('PyFER Webcam Capture', cv.cvtColor(frame_processed, cv.COLOR_RGB2BGR))

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