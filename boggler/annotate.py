import logging
import os
import string
from typing import Generator, Optional

import attr
import cv2 as cv
import numpy as np

from .data import LabeledExample, GROUND_TRUTH, LABELS, LABELS_KEYSTROKES, walk_ground_truth

LOG = logging.getLogger(__name__)

def anno_single(example: np.ndarray) -> LabeledExample:
    cv.imshow("Display window", example)
    k = cv.waitKey(0)
    if k == 27:
        raise StopIteration("User cancelled")
    else:
        c = chr(k)
        if c not in LABELS_KEYSTROKES:
            print(f"Char '{c}' (code {k}) not recognized")
        return LabeledExample(example, LABELS_KEYSTROKES[c])

def label_all(root=GROUND_TRUTH, overwrite = False):
    for im_path, gt_path in walk_ground_truth(root=root):
        if os.path.exists(gt_path) and not overwrite:
            LOG.debug(f"GT label for {im_path} already exists at {gt_path}; skipping")
        else:
            example = anno_single(cv.imread(im_path))
            with open(gt_path, "w") as gtfp:
                gtfp.write(example.label)

