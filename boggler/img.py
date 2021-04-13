import functools
import logging
import os
from pathlib import Path
from typing import List, Iterable

import cv2 as cv
import numpy as np

from boggler.data import DEBUG_OUT, ProcessedImage

PIX_CONSTANTS = {
    "open_close_kernel": 5,
    "bbox_min": 80,
    "bbox_max": 340,
}
"""All constants that scale with image size/resolution. Useful for downsampling"""
PIX_SCALE = 1.0
"""Scale applied to PIX_CONSTANTS"""

_PIX_CONSTANTS = {k: int(v * PIX_SCALE) for k, v in PIX_CONSTANTS.items()}

STORE_COMPONENTS = True
"""Store copies of bounding box contents along with ProcessedImages"""

LOG = logging.getLogger(__name__)


def bw(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def expand(img: np.ndarray) -> np.ndarray:
    """Expand dynamic range to full bit depth."""
    # NOTE: in my test images, range is already 0..255; skipping this for now
    return img


def invert(img: np.ndarray) -> np.ndarray:
    """Invert black-on-white boggle text to white-on-black."""
    return cv.bitwise_not(img)


def threshold(img: np.ndarray) -> np.ndarray:
    """Threshold a white-on-black image to binary"""
    io = img.copy()
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU, io)[1]


def open_close(img: np.ndarray) -> np.ndarray:
    """Open, then close the image. See https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html"""
    ksize = int(_PIX_CONSTANTS["open_close_kernel"])
    kernel = np.ones((ksize, ksize), np.uint8)
    opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
    return closed


def bbox(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    img = proc_img.orig
    # Tree mode ensures outer contours are visited first
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    # create nan array for overlay
    # overlay = np.empty(np.shape(img))
    # overlay[:] = np.nan
    overlay = np.copy(img)
    bboxen = []
    components = []
    added = set()
    for idx, cnt_hier in enumerate(zip(contours, hierarchy[0])):
        cnt, hier = cnt_hier
        nprev, nnext, child1, parent = hier
        if parent in added:
            LOG.debug(f"Skipping bbox {idx}, parent {parent} already in set")
            continue
        x, y, w, h = cv.boundingRect(cnt)
        if (
            _PIX_CONSTANTS["bbox_min"] < w < _PIX_CONSTANTS["bbox_max"]
            and _PIX_CONSTANTS["bbox_min"] < h < _PIX_CONSTANTS["bbox_max"]
        ):
            bboxen.append(cv.boundingRect(cnt))
            added.add(idx)
            if store_components:
                components.append(img[y : y + h, x : x + w])
            cv.rectangle(overlay, (x, y), (x + w, y + h), (200, 0, 0), 2)
    meta = {"bbox": bboxen}
    if store_components:
        meta["img_bbox"] = components
    return ProcessedImage(img, overlay, meta)

def improve_bbox(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    """Use the initial bbox hypothesis to generate an NxN grid of bboxes."""
    bboxen = proc_img.metadata["bbox"]
    proc_img.overlay = proc_img.orig.copy()
    bbox_w_h = int(1.15 * np.median([(br[2], br[3]) for br in bboxen]))
    # Set all bbox dimensions to median dimension + 15%
    def scale_around_centroid(bbin, newlen):
        x, y, w, h = bbin
        cx, cy = int(x + w/2), int(y + h/2)
        return (max(cx - newlen//2, 0), max(cy - newlen//2, 0), newlen, newlen)
    newbbs = [scale_around_centroid(bb, bbox_w_h) for bb in bboxen]
    meta = {"improved_bbox": newbbs}
    for x, y, w, h in newbbs:
        LOG.debug(f"Storing component xywh {x} {y} {w} {h}")
        cv.rectangle(proc_img.overlay, (x, y), (x + w, y + h), (200, 0, 0), 2)

    if store_components:
        meta["img_improved_bbox"] = [proc_img.orig[y : y+ h, x : x + w] for x, y, w, h in newbbs]

    proc_img.metadata.update(meta)
    return proc_img

# Aborted attempt at using pytesseract
# def chars_from_bbox(proc_img: ProcessedImage) -> ProcessedImage:
#     bbox_chars = []
#     for bbox_img in proc_img.metadata["img_bbox"]:
#         ch = pytesseract.image_to_string(bbox_img)
#         print(ch)
#         bbox_chars.append(ch)
#     proc_img.metadata.update({"bbox_chars": bbox_chars})
#     return proc_img
#


PREPROC_PIPELINE: List[callable] = [bw, expand, invert, threshold, open_close]
EXTRACT_PIPELINE: List[callable] = [bbox, improve_bbox]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


def preproc(imgs: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    for img in imgs:
        yield compose(*PREPROC_PIPELINE)(img)



def write_proc_images(imgs: Iterable[ProcessedImage], output_root: str= DEBUG_OUT):
    for i, proc in enumerate(imgs):
        this_root = os.path.join(output_root, f"{i:03d}")
        Path(this_root).mkdir(parents=True, exist_ok=True)
        cv.imwrite(os.path.join(this_root, "orig.jpg"), proc.orig)
        cv.imwrite(os.path.join(this_root, "overlay.jpg"), proc.overlay)
        for meta_key, meta_val in proc.metadata.items():
            if meta_key.startswith("img_"):
                feat_root = os.path.join(this_root, meta_key)
                Path(feat_root).mkdir(parents=True, exist_ok=True)
                for j, component in enumerate(meta_val):
                    thisout = os.path.join(feat_root, f"{j:03d}.jpg")
                    cv.imwrite(thisout, component)
        yield proc


def extract(imgs: Iterable[np.ndarray], store_components=True) -> Iterable[ProcessedImage]:
    for i, img in enumerate(imgs):
        proc = ProcessedImage(img, img, {})
        for extract_step in EXTRACT_PIPELINE:
            this_proc = extract_step(proc)
            if store_components:
                cv.imwrite(os.path.join(DEBUG_OUT, f"{i:03d}_{extract_step.__name__}.jpg"), this_proc.overlay)
            # proc.metadata.update(this_proc.metadata)
            proc = this_proc
        yield proc