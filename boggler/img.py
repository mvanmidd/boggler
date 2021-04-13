import functools
import os
from pathlib import Path
from typing import Generator, List, Iterable

import attr
import cv2 as cv
import numpy as np

IMG_IN = "img"
DEBUG_OUT = "debug"
PIX_CONSTANTS = {
    "open_close_kernel": 5,
    "bbox_min": 80,
    "bbox_max": 340,
}
"""All constants that scale with image size/resolution. Useful for downsampling"""
PIX_SCALE = 1.0
"""Scale applied to PIX_CONSTANTS"""

_PIX_CONSTANTS = {k: int(v * PIX_SCALE) for k, v in PIX_CONSTANTS.items()}


def read_images(partition="dev", ground_truth=False) -> Generator:
    for fname in os.listdir(os.path.join(IMG_IN, partition)):
        fpath = os.path.join(IMG_IN, partition, fname)
        if not fname.endswith(".jpg"):
            continue
        if ground_truth:
            raise NotImplementedError()
        yield cv.imread(cv.samples.findFile(fpath))


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
    # NOTE: in my test images, range is already 0..255; skipping this for now
    io = img.copy()
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU, io)[1]


def open_close(img: np.ndarray) -> np.ndarray:
    """Open, then close the image. See https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html"""
    ksize = int(_PIX_CONSTANTS["open_close_kernel"])
    kernel = np.ones((ksize, ksize), np.uint8)
    opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
    return closed


@attr.s()
class ProcessedImage:
    orig: np.ndarray = attr.ib()
    overlay: np.ndarray = attr.ib()
    metadata: dict = attr.ib(factory=dict)


def bbox(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    img = proc_img.orig
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    # create nan array for overlay
    # overlay = np.empty(np.shape(img))
    # overlay[:] = np.nan
    overlay = np.copy(img)
    bboxen = []
    components = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv.boundingRect(cnt)
        if (
            _PIX_CONSTANTS["bbox_min"] < w < _PIX_CONSTANTS["bbox_max"]
            and _PIX_CONSTANTS["bbox_min"] < h < _PIX_CONSTANTS["bbox_max"]
        ):
            bboxen.append(cv.boundingRect(cnt))
            if store_components:
                components.append(img[y : y + h, x : x + w])
            cv.rectangle(overlay, (x, y), (x + w, y + h), (200, 0, 0), 2)
    meta = {"bbox": bboxen}
    if store_components:
        meta["img_bbox"] = components
    return ProcessedImage(img, overlay, meta)


PREPROC_PIPELINE: List[callable] = [bw, expand, invert, threshold, open_close]
EXTRACT_PIPELINE: List[callable] = [bbox]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


def preproc(imgs: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    for img in imgs:
        yield compose(*PREPROC_PIPELINE)(img)


def extract(imgs: Iterable[np.ndarray]) -> Iterable[ProcessedImage]:
    for i, img in enumerate(imgs):
        proc = ProcessedImage(img, img, {})
        for extract_step in EXTRACT_PIPELINE:
            this_proc = extract_step(proc)
            cv.imwrite(os.path.join(DEBUG_OUT, f"{i:03d}_{extract_step.__name__}.jpg"), this_proc.overlay)
            proc.metadata.update(this_proc.metadata)
        # Write all image components
        for meta_key, meta_val in proc.metadata.items():
            Path(os.path.join(DEBUG_OUT, f"{i:03d}_debug")).mkdir(parents=True, exist_ok=True)
            if meta_key.startswith("img_"):
                for j, component in enumerate(meta_val):
                    thisout = os.path.join(DEBUG_OUT, f"{i:03d}_debug", f"{meta_key}_{j:03d}.jpg")
                    cv.imwrite(thisout, component)

        yield proc