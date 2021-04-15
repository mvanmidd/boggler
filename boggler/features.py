import logging
import random
from typing import Iterable, Generator

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .data import read_ground_truth, LabeledExample, LABELS, GROUND_TRUTH

FEATURE_SIZE = (32, 32)

TRAIN_EPOCHS = 15

NUM_CLASSES = len(LABELS)

LOG = logging.getLogger(__name__)

OVERSAMPLE = {l: 2 for l in LABELS}
# hard cases
OVERSAMPLE.update(
    {"in": 8, "he": 8, "th": 8, "er": 8, "qu": 4, "an": 8, "!": 2,}
)
# stuff that's underrepresented in my data for no particular reason
OVERSAMPLE.update(
    {"f": 4, "j": 4, "v": 4, "y": 6}
)


def scale(img: np.ndarray) -> np.ndarray:
    return cv.resize(img, FEATURE_SIZE)


def rotate(img: np.ndarray) -> Generator:
    yield img
    yield cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    yield cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    yield cv.rotate(img, cv.ROTATE_180)


def perspective(img: np.ndarray, magnitude: float = 0.2) -> np.ndarray:
    return img


def onehot_label(label):
    oh = np.zeros([NUM_CLASSES])
    oh[LABELS.index(label)] = 1
    return oh

#
# def serialize_example(img, label):
#     tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#

def randscale_crop(img, min_visible=0.85) -> np.ndarray:
    # scale_factor = 1.0 + random.random() * (1.0 - min_visible)
    scale_factor = np.random.uniform(1.0, 1.0 / min_visible, [2])
    orig_shape = np.array(np.shape(img))
    new_shape = orig_shape * scale_factor
    # I am bad at np
    new_shape = np.array([int(new_shape[0]), int(new_shape[1])])
    LOG.debug(f"Scale factor f{scale_factor}. {orig_shape} -> {new_shape}")
    newim = cv.resize(img, tuple(new_shape))
    newxy = (new_shape - orig_shape) * np.random.uniform(size=[2])
    newxy = tuple(int(d) for d in newxy)
    LOG.debug(f"New origin: {newxy}")
    return newim[newxy[0] : newxy[0] + orig_shape[0], newxy[1] : newxy[1] + orig_shape[1]]


def feat_permutations(img: np.ndarray, oversample=1) -> Generator:
    for rot in rotate(img):
        yield scale(rot.copy())
        for _ in range(oversample):
            yield scale(randscale_crop(rot.copy()))


def iter_feats(root=GROUND_TRUTH, partition="", oversample=True):
    for img, gt in read_ground_truth(root, partition):
        if gt != "?":
            for feat in feat_permutations(img, OVERSAMPLE[gt] if oversample else 1):
                yield feat, gt


def batch(seq, bsize=10):
    i = 0
    out = []
    while i < len(seq):
        out += seq[i : i + bsize]
        i += bsize
    return out


#### TODO: Move below to models.py?
def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model

def cnn_model_orig():
    model = models.Sequential()
    model.add(layers.Conv2D(40, (3, 3), activation="relu", input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES))
    return model


def partitioned_feats_labels(partition="TRAIN", oversample=True):
    allfeats = list(iter_feats(partition=partition, oversample=oversample))
    random.shuffle(allfeats)
    images, labels = (
        np.array([np.expand_dims(i, -1) for i, l in allfeats]),
        np.array([onehot_label(l) for i, l in allfeats]),
    )
    return images, labels


def train():
    model = cnn_model()
    print(model.summary())
    train_img, train_label = partitioned_feats_labels(partition="TRAIN")
    val_img, val_label = partitioned_feats_labels(partition="TEST", oversample=False)
    model.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

    history = model.fit(train_img, train_label, epochs=TRAIN_EPOCHS, validation_data=(val_img, val_label))

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()

    test_loss, test_acc = model.evaluate(val_img, val_label, verbose=2)
    return model, history, val_img, val_label
