import cv2 as cv
import numpy as np


def fer(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)


def affectnet(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)
