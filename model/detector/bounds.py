from typing import Sequence

import cv2 as cv
import numpy as np
from ultralytics.engine.results import Results


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

    def __str__(self):
        return f'BoundingBox[class={self.name} confidence={self.confidence} start={self.start} end={self.end}]'

    @staticmethod
    def wrap(results: Results) -> list:
        """
        Converts a YOLOv8 Results object into a friendlier list of BoundingBox objects. :)
        :param results: Results object obtained by applying the YOLOv8 model to an image.
        :return: List of bounding boxes for objects found by the YOLOv8 model.
        """
        lst = []

        boxes = results.boxes
        cls = boxes.cls.tolist()
        conf = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()

        for i in range(len(cls)):
            name = results.names[cls[i]]
            start = (int(xyxy[i][0]), int(xyxy[i][1]))
            end = (int(xyxy[i][2]), int(xyxy[i][3]))
            confidence = conf[i]
            lst.append(BoundingBox(name, start, end, confidence))

        return lst


def annotated(
        image: np.ndarray | str,
        bounding_boxes: list[BoundingBox],
        include_title: bool = True,
        colour: Sequence[int] = (255, 0, 0)
) -> np.ndarray:
    """
    Annotates an image with the bounding boxes of the objects present in the image.
    :param image: The image to annotate, or the path to the file it's stored in.
    :param bounding_boxes: A list of bounding boxes.
    :param include_title: True if the class name should be drawn along with the bounding box; False otherwise.
    :param colour: The colour to use when drawing the bounding boxes.
    :return: An annotated image.
    """
    img = image.copy() if isinstance(image, np.ndarray) else cv.imread(image)
    for box in bounding_boxes:
        if include_title:
            text_pos = (box.start[0], box.start[1] - 10)
            title = f'{box.name} ({box.confidence})'
            cv.putText(img, title, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2, cv.LINE_AA)
        cv.rectangle(img, box.start, box.end, colour, thickness=2)
    return img
