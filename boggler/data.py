import os
import string
from typing import Generator, Optional

import attr
import cv2 as cv
import numpy as np

IMG_IN = "img"
DEBUG_OUT = "debug"
FEATURES = "features"


def read_images(root=IMG_IN, partition="dev", ground_truth=False) -> Generator:
    for fname in os.listdir(os.path.join(root, partition)):
        fpath = os.path.join(root, partition, fname)
        if not fname.endswith(".jpg"):
            continue
        if ground_truth:
            raise NotImplementedError()
        yield cv.imread(fpath)



@attr.s()
class ProcessedImage:
    orig: np.ndarray = attr.ib()
    overlay: np.ndarray = attr.ib()
    metadata: dict = attr.ib(factory=dict)


LABELS = ({"er", "qu", "!", "?"} | set(string.ascii_lowercase)) - {"q"}
LABELS_KEYSTROKES = {"E": "er", "q": "qu", "!": "!", "?": "?"}
LABELS_KEYSTROKES.update({k: k for k in string.ascii_lowercase if k != "q"})

@attr.s()
class LabeledExample:
    img: np.ndarray = attr.ib()
    label: Optional[str] = attr.ib(default=None)

    @label.validator
    def _validate_label(self, _, value):
        if not (value is None or value in LABELS):
            raise ValueError(f"Unknown label {value}")
