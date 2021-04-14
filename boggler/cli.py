import click
import cv2 as cv
import logging

from boggler.img import preproc, extract, write_proc_images
from boggler.annotate import label_all
from boggler.data import read_images, GROUND_TRUTH
from boggler.features import iter_feats

LOG = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

@cli.command()
@click.option("-i", "--interactive", is_flag=True, default=False)
def seg(interactive):
    origim = read_images()
    for pimg in write_proc_images(extract(preproc(read_images()))):
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
    for feat, label in iter_feats(ground_truth_dir):
        if interactive:
            cv.imshow("Display window", feat)
            k = cv.waitKey(0)



