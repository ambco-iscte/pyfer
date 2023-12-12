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
### Using Pre-Trained Models

### Defining your own Models
#### Object Detection


#### Facial Expression Classification

<br>

## Example
The following is an example of applying PyFER to a single image.
```python
# Load detector and classifier models
detector = FaceDetector(YOLO('path/to/best.pt'))
classifier = EmotionClassifier(load_model('path/to/tensorflow/model'))

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