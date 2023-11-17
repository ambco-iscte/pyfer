import cv2 as cv
import numpy as np

from typing import Sequence
from xml.etree import ElementTree

# PyFER - Group 02
# Afonso Caniço     92494
# Gustavo Ferreira  92888
# Samuel Correia    92619

class BoundingBox:
    """
    Class representing a bounding box for an object.
    """
    def __init__(self, name: str, start: (int, int), end: (int, int), confidence: float):
        if start[0] > end[0] or start[1] > end[1]:
            raise Exception("Starting point (top left) must be smaller than or equal to the end point (bottom right).")
        self.name = name
        self.start = start
        self.end = end
        self.confidence = round(max(0.0, min(confidence, 1.0)), 4)
        self.width = end[0] - start[0]
        self.height = end[1] - start[1]
        self.size = (self.width, self.height)



