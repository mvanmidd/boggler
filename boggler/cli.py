import json
from collections import Counter

import click
import cv2 as cv
import logging

from boggler.img import preproc, extract, write_proc_images
from boggler.annotate import label_all
from boggler.data import read_images, GROUND_TRUTH
from boggler.features import iter_feats, partitioned_feats_labels

LOG = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

@cli.command()
@click.option("-i", "--interactive", is_flag=True, default=False)
@click.option("-s", "--store-components", is_flag=True, default=False)
def seg(interactive, store_components):
    for fname, pimg in write_proc_images(extract(preproc(read_images(), store_components))):
        img = pimg.img
        if interactive:
            LOG.info(f"{len(pimg.metadata['improved_bbox'])} bboxes found")
            cv.imshow("Display window", img)
            k = cv.waitKey(0)
            print(k)

@cli.command()
@click.option("-g", "--ground-truth-dir", default=GROUND_TRUTH)
def annotate(ground_truth_dir):
    label_all(ground_truth_dir)


@cli.command()
@click.option("-g", "--ground-truth-dir", default=GROUND_TRUTH)
@click.option("-i", "--interactive", is_flag=True, default=False)
def genfeats(ground_truth_dir, interactive):
    alltrain = list(iter_feats(partition="TRAIN"))
    trainfeats, trainlabels = [f for f, l in alltrain], [l for f, l in alltrain]
    alltest = list(iter_feats(partition="TEST"))
    testfeats, testlabels = [f for f, l in alltest], [l for f, l in alltest]
    trainstats = Counter(trainlabels)
    teststats = Counter(testlabels)
    print(f"TRAIN stats:\n{json.dumps(trainstats,sort_keys=True, indent=2)}\n")
    print(f"TEST stats:\n{json.dumps(teststats,sort_keys=True, indent=2)}\n")
    if interactive:
        for feat in trainfeats[:20]:
            cv.imshow("Display window", feat)
            k = cv.waitKey(0)
        for feat in testfeats[:20]:
            cv.imshow("Display window", feat)
            k = cv.waitKey(0)




