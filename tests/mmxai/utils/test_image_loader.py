from mmxai.utils.image_loader import loadImage

from PIL import Image, ImageChops
import numpy as np
import requests

url = "https://www.iqmetrix.com/hubfs/Meme%2021.jpg"
img = Image.open(requests.get(url, stream=True).raw)

def testCanPassThroughPILImage():
    img_new = loadImage(img)
    checkTwoImageAreSame(img, img_new)


def testCanImportImageFromNumpyArray():
    img_array = np.array(img)
    img_new = loadImage(img_array)

    checkTwoImageAreSame(img, img_new)  

def testCanImportImageFromURL():
    img_new = loadImage(url)

    checkTwoImageAreSame(img, img_new)

def testCanImportImageFromLocalPath():
    img_new = loadImage("tests/mmxai/text_removal/test_img.jpg")

    checkTwoImageAreSame(img, img_new)

def checkAbortWhenInputIsUnsupported():
    try:
        img = loadImage(123456789.0)
    except:
        assert True
    else:
        assert False

def checkTwoImageAreSame(img1, img2):
    assert isinstance(img1, Image.Image)
    assert isinstance(img2, Image.Image)

    diff = ImageChops.difference(img1, img2)
    assert diff.getbbox() is None


if __name__ == "__main__":
    testCanPassThroughPILImage()
    testCanImportImageFromNumpyArray()
    testCanImportImageFromURL()
    testCanImportImageFromLocalPath()
    checkAbortWhenInputIsUnsupported()