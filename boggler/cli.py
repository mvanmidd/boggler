import click
import cv2 as cv

from boggler.img import read_images, preproc, extract


@click.command()
@click.option("-i", "--interactive", is_flag=True, default=False)
def boggler(interactive):
    for pimg in extract(preproc(read_images())):
        img = pimg.overlay
        if interactive:
            cv.imshow("Display window", img)
            k = cv.waitKey(0)


def main():
    boggler()
