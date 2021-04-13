import click
import cv2 as cv
import logging

from boggler.img import preproc, extract, write_proc_images
from boggler.annotate import label_all
from boggler.data import read_images, FEATURES


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

@cli.command()
@click.option("-i", "--interactive", is_flag=True, default=False)
def seg(interactive):
    origim = read_images()
    for pimg in write_proc_images(extract(preproc(read_images()))):
        img = pimg.overlay
        if interactive:
            cv.imshow("Display window", img)
            k = cv.waitKey(0)
            print(k)

@cli.command()
@click.option("-f", "--feats-dir", default=FEATURES)
def annotate(feats_dir):
    label_all(feats_dir)

