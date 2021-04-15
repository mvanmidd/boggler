import functools
import logging
import os
from itertools import combinations, product
from pathlib import Path
from typing import List, Iterable, Tuple

import cv2 as cv
import numpy as np

from boggler.data import DEBUG_OUT, ProcessedImage, FEATURES

PIX_CONSTANTS = {
    "open_close_kernel": 5,
    "bbox_min": 140,
    "bbox_max": 640,
    "local_contrast_window_size": 500,
    "median_blur_kernel_size": 19,
}
"""All constants that scale with image size/resolution. Useful for downsampling"""
PIX_SCALE = 1.0
"""Scale applied to PIX_CONSTANTS"""

_PIX_CONSTANTS = {k: int(v * PIX_SCALE) for k, v in PIX_CONSTANTS.items()}

BOARD_SIZES = (4, 6)
"""Valid nxn board sizes"""

DETECT_BOARD = False
"""Experimental: first find bounding box of entire board."""

HOUGHLINES = False
"""Experimental: lines-based segmenter. Works poorly."""

STORE_COMPONENTS = True
"""Store copies of bounding box contents along with ProcessedImages"""

LOG = logging.getLogger(__name__)


def grayscale(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def crop_square(img: np.ndarray) -> np.ndarray:
    """Crop an image to sqaure along the shorter dimension, centering along longer dimension."""
    s = img.shape
    if s[0] > s[1]:
        offset = (s[0] - s[1]) // 2
        img = img[offset : offset + s[1], :]
    if s[0] < s[1]:
        offset = (s[1] - s[0]) // 2
        img = img[:, offset : offset + s[0]]
    return img


def median_blur(img: np.ndarray) -> np.ndarray:
    return cv.medianBlur(img, _PIX_CONSTANTS["median_blur_kernel_size"])


def adaptive_contrast(img: np.ndarray) -> np.ndarray:
    """Apply local contrast enhancement."""
    # create a CLAHE object (Arguments are optional).
    winsize = _PIX_CONSTANTS["local_contrast_window_size"]
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(winsize, winsize))
    cl1 = clahe.apply(img)
    return cl1


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


def bbox_board(proc_img: ProcessedImage, store_components=True) -> ProcessedImage:
    img = proc_img.orig
    # Tree mode ensures outer contours are visited first
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    # create nan array for overlay
    # overlay = np.empty(np.shape(img))
    # overlay[:] = np.nan
    overlay = np.copy(img)
    imsize = img.shape
    bboxen = []
    components = []
    added = set()
    for idx, cnt_hier in enumerate(zip(contours, hierarchy[0])):
        cnt, hier = cnt_hier
        nprev, nnext, child1, parent = hier
        # if parent in added:
        #     LOG.debug(f"Skipping bbox {idx}, parent {parent} already in set")
        #     continue
        x, y, w, h = cv.boundingRect(cnt)
        if True or (w > imsize[1] * 0.5 and h > imsize[0] * 0.4):
            bboxen.append(cv.boundingRect(cnt))
            added.add(idx)
            if store_components:
                components.append(img[y : y + h, x : x + w])
            cv.rectangle(overlay, (x, y), (x + w, y + h), (200, 0, 0), 8)
    meta = {"bbox": bboxen}
    if store_components:
        meta["img_bbox"] = components
    return ProcessedImage(img, overlay, meta)


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


def outlier_reject_elliptic(proc_img: ProcessedImage) -> ProcessedImage:
    """Didn't work very well."""
    from sklearn.covariance import EllipticEnvelope

    bboxen = proc_img.metadata["bbox"]
    bbox_xy = [(x, y) for x, y, _, __ in bboxen]
    cov = EllipticEnvelope().fit(bbox_xy)
    fit = cov.predict(bbox_xy)
    LOG.debug(f"Outlier predictions: {fit}")
    LOG.debug(f"Not performing outlier rejection because it is not yet trustworthy. Maybe we don't need it.")
    return proc_img


def outlier_reject_manhattan(proc_img: ProcessedImage) -> ProcessedImage:
    from sklearn.neighbors import LocalOutlierFactor

    bboxen = proc_img.metadata["bbox"]
    # bbox_xy = [(x, y) for x, y, _, __ in bboxen]
    bbox_xy = bboxen
    clf = LocalOutlierFactor(n_neighbors=10, metric="l1")
    fit = clf.fit_predict(bbox_xy)
    inlier_bboxen = [bboxen[i] for i, v in enumerate(fit) if v == 1]
    outlier_bboxen = [bboxen[i] for i, v in enumerate(fit) if v == -1]
    LOG.debug(f"Removed {len(outlier_bboxen)} outliers.")
    proc_img.metadata["bbox"] = inlier_bboxen
    proc_img.img = proc_img.orig.copy()
    for x, y, w, h in inlier_bboxen:
        cv.rectangle(proc_img.img, (x, y), (x + w, y + h), (200, 0, 0), 8)
    for x, y, w, h in outlier_bboxen:
        cv.rectangle(proc_img.img, (x, y), (x + w, y + h), (200, 0, 0), 14)
    return proc_img


def square_crop_from_bbox_bounds(proc_img: ProcessedImage) -> ProcessedImage:
    bboxen = proc_img.metadata["improved_bbox"]
    l, r, t, b = [
        min(p[0] for p in bboxen),
        max(p[0] + p[2] for p in bboxen),
        min(p[1] for p in bboxen),
        max(p[1] + p[3] for p in bboxen),
    ]
    bot_right = max(bboxen, key=lambda p: p[0] + p[1] + p[2] + p[3])
    bot_left = max(bboxen, key=lambda p: p[1] + p[3] - p[0])
    top_right = max(bboxen, key=lambda p: p[0] + p[2] - p[1])
    cv.rectangle(proc_img.img, (l, t), (r, b), (200, 0, 0), 12)
    top_left = (l, t)
    bot_right = (r, b)
    bot_left = (l, b)
    top_right = (r, t)
    mindim = min(np.shape(proc_img.orig))
    tl, br, tr, bl = (0, 0), (mindim, mindim), (mindim, 0), (0, mindim)
    transorm = cv.getPerspectiveTransform(
        np.float32((top_left, top_right, bot_left, bot_right)), np.float32((tl, tr, bl, br))
    )
    transformed = cv.warpPerspective(proc_img.orig, transorm, (mindim, mindim))
    proc_img.img = transformed
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
    nguess = min(BOARD_SIZES, key=lambda n: abs(n * n - len(proc_img.metadata["improved_bbox"])))
    LOG.info(f"Guessing that this is {nguess} x {nguess} board")
    imsize = np.shape(proc_img.img)[0]  # must be square image by this point
    gridspace = imsize // nguess
    # Write our features
    new_bbox = []
    new_bbim = []
    for x, y in product(list(range(nguess)), list(range(nguess))):
        new_bbim.append(
            proc_img.img[x * gridspace : (x + 1) * gridspace, y * gridspace : (y + 1) * gridspace].copy()
        )
        proc_img.metadata[f"img_feat_{y}{x}"] = [
            proc_img.img[x * gridspace : (x + 1) * gridspace, y * gridspace : (y + 1) * gridspace].copy()
        ]
        new_bbox.append((x, y, gridspace, gridspace))
    proc_img.metadata["grid_bbox"] = new_bbox
    proc_img.metadata["img_grid_bbox"] = new_bbim

    # Generate a grid overlay
    for x in range(0, imsize, gridspace):
        cv.line(proc_img.img, (x, 0), (x, imsize), color=(200, 0, 0), thickness=6)
    for y in range(0, imsize, gridspace):
        cv.line(proc_img.img, (0, y), (imsize, y), color=(200, 0, 0), thickness=6)
    return proc_img

def floodfill_outside(proc_img: ProcessedImage) -> ProcessedImage:
    """Floodfill starting at all white points along outside of image"""
    grid_bb = proc_img.metadata["img_grid_bbox"]
    bbsize = proc_img.metadata["img_grid_bbox"][0].shape[0]
    print(grid_bb[0].shape)
    for bbi in grid_bb:
        ffilled = bbi
        for ffpoint in range(bbsize):
            for pt in [(0, ffpoint), (bbsize-1, ffpoint), (ffpoint, bbsize-1), (ffpoint, 0)]:
                if ffilled[pt] != 0:
                    print(f"Filling from {pt}")
                    cv.floodFill(ffilled, None, (pt[1], pt[0]), 0, flags=8)
        cv.imshow("display", ffilled)
        cv.waitKey(0)
    return proc_img



def houghlines(proc_img: ProcessedImage) -> ProcessedImage:
    imsi = proc_img.img.shape
    downs = cv.resize(proc_img.img, (int(imsi[1] * 0.3), int(imsi[0] * 0.3)))

    minLineLength = 100
    maxLineGap = 20
    edges = cv.Canny(downs, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 0.1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv.line(downs, (x1, y1), (x2, y2), (200, 5, 0), 30)
    #
    # edges = cv.Canny(downs, 50, 150, apertureSize=7)
    # lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #
    #     cv.line(downs, (x1, y1), (x2, y2), (0, 0, 255), 2)
    proc_img.img = downs
    return proc_img


outlier_reject = outlier_reject_manhattan

if HOUGHLINES:
    PREPROC_PIPELINE: List[callable] = [grayscale, median_blur, adaptive_contrast, invert]
    PREPROC_PIPELINE: List[callable] = [grayscale]
    EXTRACT_PIPELINE: List[callable] = [
        houghlines,
    ]
elif DETECT_BOARD:
    PREPROC_PIPELINE: List[callable] = [
        grayscale,
        median_blur,
        adaptive_contrast,
        invert,
        threshold,
        open_close,
    ]
    EXTRACT_PIPELINE: List[callable] = [
        bbox_board,
    ]
else:
    PREPROC_PIPELINE: List[callable] = [
        grayscale,
        crop_square,
        median_blur,
        # Too aggressive
        # adaptive_contrast,
        invert,
        threshold,
        open_close,
    ]
    EXTRACT_PIPELINE: List[callable] = [
        bbox,
        improve_bbox,
        combine_overlapping_bbox,
        outlier_reject,
        # reshape corrects perspective too, but it's a bit aggressive
        # reshape_from_bbox_bounds,
        square_crop_from_bbox_bounds,
        gridify,
        floodfill_outside,
    ]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


def preproc(imgs: Iterable[Tuple[str, np.ndarray]], store_components=False) -> Iterable[Tuple[str, np.ndarray]]:
    if store_components:
        yield from _preproc_debug(imgs)
    else:
        for img in imgs:
            yield compose(*PREPROC_PIPELINE)(img)


def _preproc_debug(imgs: Iterable[Tuple[str, np.ndarray]], output_root: str = DEBUG_OUT) -> Iterable[np.ndarray]:
    for i, imgf in enumerate(imgs):
        fname, img = imgf
        this_root = os.path.join(output_root, f"{i:03d}", "preprocessing")
        Path(this_root).mkdir(parents=True, exist_ok=True)
        cv.imwrite(os.path.join(this_root, "orig.jpg"), img)
        for j, preproc_step in enumerate(PREPROC_PIPELINE):
            img = preproc_step(img)
            cv.imwrite(os.path.join(this_root, f"preproc_{j:02d}_{preproc_step.__name__}.jpg"), img)
        yield fname, img


def write_proc_images(
    imgs: Iterable[Tuple[str, ProcessedImage]], feat_output_root: str = FEATURES, debug_output_root: str = DEBUG_OUT
) -> Tuple[str, ProcessedImage]:
    for i, procf in enumerate(imgs):
        fname, proc = procf
        debug_root = os.path.join(debug_output_root, f"{i:03d}_{fname}")
        feat_root = os.path.join(feat_output_root, f"{i:03d}_{fname}")
        Path(debug_root).mkdir(parents=True, exist_ok=True)
        Path(feat_root).mkdir(parents=True, exist_ok=True)
        cv.imwrite(os.path.join(debug_root, "orig.jpg"), proc.orig)
        cv.imwrite(os.path.join(debug_root, "processed.jpg"), proc.img)
        for meta_key, meta_val in proc.metadata.items():
            if meta_key.startswith("img_feat_"):
                this_feat_path = os.path.join(feat_root, f"{meta_key}.jpg")
                assert len(meta_val) == 1, "Only one output feature per key allowed!"
                cv.imwrite(this_feat_path, meta_val[0])
            elif meta_key.startswith("img_"):
                debug_feat_root = os.path.join(debug_root, meta_key)
                Path(debug_feat_root).mkdir(parents=True, exist_ok=True)
                for j, component in enumerate(meta_val):
                    thisout = os.path.join(debug_feat_root, f"{j:03d}.jpg")
                    cv.imwrite(thisout, component)
        yield fname, proc


def extract(imgs: Iterable[Tuple[str, np.ndarray]], store_components=True) -> Iterable[Tuple[str, ProcessedImage]]:
    for i, imgf in enumerate(imgs):
        fname, img = imgf
        proc = ProcessedImage(img, img, {})
        for j, extract_step in enumerate(EXTRACT_PIPELINE):
            this_proc = extract_step(proc)
            if store_components:
                cv.imwrite(
                    os.path.join(DEBUG_OUT, f"{i:03d}_{fname}_{j:02d}_{extract_step.__name__}.jpg"), this_proc.img
                )
            # proc.metadata.update(this_proc.metadata)
            proc = this_proc
        yield fname, proc
