import os
import string
from typing import Generator, Optional

import attr
import cv2 as cv
import numpy as np

IMG_IN = "img"
DEBUG_OUT = "debug"
GROUND_TRUTH = "groundtruth"
FEATURES = "features"


def read_images(root=IMG_IN, partition="", ground_truth=False) -> Generator:
    for dirpath, dnames, fnames in os.walk(os.path.join(root, partition)):
        for fname in fnames:
            fpath = os.path.join(dirpath, fname)
            if not fname.endswith(".jpg"):
                continue
            if ground_truth:
                raise NotImplementedError()
            yield cv.imread(fpath)



@attr.s()
class ProcessedImage:
    orig: np.ndarray = attr.ib()
    img: np.ndarray = attr.ib()
    metadata: dict = attr.ib(factory=dict)


LABELS = ["in", "he", "th", "er", "qu", "an", "!", "?"]  + list(c for c in string.ascii_lowercase if c != "q")
LABELS_KEYSTROKES = {"I": "in", "H": "he", "T": "th", "E": "er", "Q": "qu", "A": "an", "!": "!", "?": "?"}
LABELS_KEYSTROKES.update({k: k for k in string.ascii_lowercase if k != "q"})

@attr.s()
class LabeledExample:
    img: np.ndarray = attr.ib()
    label: Optional[str] = attr.ib(default=None)

    @label.validator
    def _validate_label(self, _, value):
        if not (value is None or value in LABELS):
            raise ValueError(f"Unknown label {value}")

def walk_ground_truth(root=GROUND_TRUTH):
    for dirpath, dnames, fnames in os.walk(root):
        for f in fnames:
            if f.endswith(".jpg"):
                gt_name = os.path.splitext(f)[0] + ".label"
                im_path = os.path.join(dirpath, f)
                gt_path = os.path.join(dirpath, gt_name)
                yield (im_path, gt_path)


def read_ground_truth(root=GROUND_TRUTH):
    for im_path, gt_path in walk_ground_truth(root):
        gt = None
        if os.path.exists(gt_path):
            with open(gt_path) as gtin:
                gt = gtin.read()
        img = cv.cvtColor(cv.imread(im_path), cv.COLOR_BGR2GRAY)
        img= cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        yield img, gt