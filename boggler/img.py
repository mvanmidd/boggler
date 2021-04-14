import functools
import logging
import os
from itertools import combinations, product
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


def combine_overlapping_bbox(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    """Yeah, it's probably n^2. Also mutates proc_img.metadata["bbox"]. Watch out.

    Horrendous duplicate-the-list hack courtesy of
    https://answers.opencv.org/question/204530/merge-overlapping-rectangles/ and a bunch of sketchy stackoverflows.

    """
    bbox = proc_img.metadata["bbox"]
    # magic number epsilon, here be dragons. eps 0 -> no grouping, eps inf -> single group
    grouped_bbox, weights = cv.groupRectangles(bbox + bbox, groupThreshold=1, eps=0.2)
    proc_img.img = proc_img.orig.copy()
    for x, y, w, h in grouped_bbox:
        cv.rectangle(proc_img.img, (x, y), (x + w, y + h), (200, 0, 0), 4)
    LOG.info(f"Combined {len(bbox) - len(grouped_bbox)} duplicate bounding boxes")
    proc_img.metadata["bbox"] = grouped_bbox
    return proc_img


def improve_bbox(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    """Use the initial bbox hypothesis to generate an NxN grid of bboxes."""
    bboxen = proc_img.metadata["bbox"]
    proc_img.img = proc_img.orig.copy()
    bbox_w_h = int(1.15 * np.median([(br[2], br[3]) for br in bboxen]))
    # Set all bbox dimensions to median dimension + 15%
    def scale_around_centroid(bbin, newlen):
        x, y, w, h = bbin
        cx, cy = int(x + w / 2), int(y + h / 2)
        return (max(cx - newlen // 2, 0), max(cy - newlen // 2, 0), newlen, newlen)

    newbbs = [scale_around_centroid(bb, bbox_w_h) for bb in bboxen]
    meta = {"improved_bbox": newbbs}
    for x, y, w, h in newbbs:
        LOG.debug(f"Storing component xywh {x} {y} {w} {h}")
        cv.rectangle(proc_img.img, (x, y), (x + w, y + h), (200, 0, 0), 2)

    if store_components:
        meta["img_improved_bbox"] = [proc_img.orig[y : y + h, x : x + w] for x, y, w, h in newbbs]

    proc_img.metadata.update(meta)
    return proc_img


def outlier_reject(proc_img: ProcessedImage) -> ProcessedImage:
    from sklearn.covariance import EllipticEnvelope

    bboxen = proc_img.metadata["bbox"]
    bbox_xy = [(x, y) for x, y, _, __ in bboxen]
    cov = EllipticEnvelope().fit(bbox_xy)
    fit = cov.predict(bbox_xy)
    LOG.debug(f"Outlier predictions: {fit}")
    LOG.debug(f"Not performing outlier rejection because it is not yet trustworthy. Maybe we don't need it.")
    return proc_img


def reshape_from_bbox_bounds(proc_img: ProcessedImage) -> ProcessedImage:
    bboxen = proc_img.metadata["improved_bbox"]
    top_left = min(bboxen, key=lambda p: p[0] + p[1])
    bot_right = max(bboxen, key=lambda p: p[0] + p[1] + p[2] + p[3])
    bot_left = max(bboxen, key=lambda p: p[1] + p[3] - p[0])
    top_right = max(bboxen, key=lambda p: p[0] + p[2] - p[1])
    for x, y, w, h in top_left, bot_right, bot_left, top_right:
        cv.rectangle(proc_img.img, (x, y), (x + w, y + h), (200, 0, 0), 8)
    top_left = (top_left[0], top_left[1])
    bot_right = (bot_right[0] + bot_right[2], bot_right[1] + bot_right[3])
    bot_left = (bot_left[0], bot_left[1] + bot_left[3])
    top_right = (top_right[0] + top_right[2], top_right[1])
    mindim = min(np.shape(proc_img.orig))
    tl, br, tr, bl = (0, 0), (mindim, mindim), (mindim, 0), (0, mindim)
    transorm = cv.getPerspectiveTransform(
        np.float32((top_left, top_right, bot_left, bot_right)), np.float32((tl, tr, bl, br))
    )
    transformed = cv.warpPerspective(proc_img.orig, transorm, (mindim, mindim))
    proc_img.img = transformed
    return proc_img
    # LOG.debug(f"Bounds: ")


def gridify(proc_img: ProcessedImage) -> ProcessedImage:
    """Take a rectified + cropped ProcessedImage and fit an NxN grid, N == 4, 5, or 6 based on num bboxes."""
    nguess = min([4, 5, 6], key=lambda n: abs(n * n - len(proc_img.metadata["improved_bbox"])))
    LOG.info(f"Guessing that this is {nguess} x {nguess} board")
    imsize = np.shape(proc_img.img)[0]  # must be square image by this point
    gridspace = imsize // nguess
    # Write our features
    for x, y in product(list(range(nguess)), list(range(nguess))):
        proc_img.metadata[f"img_feat_{y}{x}"] = [proc_img.img[
            x * gridspace : (x + 1) * gridspace, y * gridspace : (y + 1) * gridspace
        ].copy()]
    # Generate a grid overlay
    for x in range(0, imsize, gridspace):
        cv.line(proc_img.img, (x, 0), (x, imsize), color=(200, 0, 0), thickness=6)
    for y in range(0, imsize, gridspace):
        cv.line(proc_img.img, (0, y), (imsize, y), color=(200, 0, 0), thickness=6)
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
EXTRACT_PIPELINE: List[callable] = [
    bbox,
    outlier_reject,
    combine_overlapping_bbox,
    improve_bbox,
    reshape_from_bbox_bounds,
    gridify,
]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


def preproc(imgs: Iterable[np.ndarray], store_components=False) -> Iterable[np.ndarray]:
    if store_components:
        yield from _preproc_debug(imgs)
    else:
        for img in imgs:
            yield compose(*PREPROC_PIPELINE)(img)


def _preproc_debug(imgs: Iterable[np.ndarray], output_root: str = DEBUG_OUT) -> Iterable[np.ndarray]:
    for i, img in enumerate(imgs):
        this_root = os.path.join(output_root, f"{i:03d}", "preprocessing")
        Path(this_root).mkdir(parents=True, exist_ok=True)
        cv.imwrite(os.path.join(this_root, "orig.jpg"), img)
        for j, preproc_step in enumerate(PREPROC_PIPELINE):
            img = preproc_step(img)
            cv.imwrite(os.path.join(this_root, f"preproc_{j:02d}_{preproc_step.__name__}.jpg"), img)
        yield img


def write_proc_images(imgs: Iterable[ProcessedImage], output_root: str = DEBUG_OUT):
    for i, proc in enumerate(imgs):
        this_root = os.path.join(output_root, f"{i:03d}")
        Path(this_root).mkdir(parents=True, exist_ok=True)
        cv.imwrite(os.path.join(this_root, "orig.jpg"), proc.orig)
        cv.imwrite(os.path.join(this_root, "processed.jpg"), proc.img)
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
        for j, extract_step in enumerate(EXTRACT_PIPELINE):
            this_proc = extract_step(proc)
            if store_components:
                cv.imwrite(
                    os.path.join(DEBUG_OUT, f"{i:03d}_{j:02d}_{extract_step.__name__}.jpg"), this_proc.img
                )
            # proc.metadata.update(this_proc.metadata)
            proc = this_proc
        yield proc
