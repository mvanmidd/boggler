import click
import cv2 as cv
import logging

from boggler.img import read_images, preproc, extract, write_proc_images


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

@cli.command()
def annotate():
    raise NotImplementedError()

