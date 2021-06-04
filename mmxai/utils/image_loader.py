from PIL import Image
import numpy as np
import requests
import sys


def loadImage(img):
    if isinstance(img, Image.Image):
        pass
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, str):
        if img.startswith("https"):
            img = Image.open(requests.get(img, stream=True).raw)
        else:
            img = Image.open(img)
    else: # pragma: no cover
        sys.exit("ERROR: Unsupported img type. Abort")

    return img
