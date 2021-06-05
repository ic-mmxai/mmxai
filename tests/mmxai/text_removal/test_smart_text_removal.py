from mmxai.text_removal.smart_text_removal import SmartTextRemover

from PIL import Image
import numpy as np
import requests
import cv2 as cv

url = "https://www.iqmetrix.com/hubfs/Meme%2021.jpg"
img = Image.open(requests.get(url, stream=True).raw)

def testCanInitSmartTextRemover():
    remover = SmartTextRemover()
    assert isinstance(remover.detector, cv.dnn_Net)

def testConvertRgbToBgr():
    remover = initRemover()
    
    img_array = np.array(img)
    img_BGR = remover.convertRgbToBgr(img_array)

    assert np.array_equal(img_array[:,:,0], img_BGR[:,:,-1])
    assert np.array_equal(img_array[:,:,1], img_BGR[:,:,1])
    assert np.array_equal(img_array[:,:,-1], img_BGR[:,:,0])

def testFindGradientAndYIntersec():
    remover = initRemover()

    p1 = np.array([-1, 3])
    p2 = np.array([3, 11])

    gradient_1, y_intersec_1 = remover.findGradientAndYIntersec(p1, p2)
    gradient_2, y_intersec_2 = remover.findGradientAndYIntersec(p1, p2)

    assert gradient_1 == gradient_2 == 2.0
    assert y_intersec_1 == y_intersec_2 == 5.0

    p3 = np.array([2, 3])
    p4 = np.array([8, 6])

    gradient_3, y_intersec_3 = remover.findGradientAndYIntersec(p3, p4)
    gradient_4, y_intersec_4 = remover.findGradientAndYIntersec(p3, p4)

    assert gradient_3 == gradient_4 == 0.5
    assert y_intersec_3 == y_intersec_4 == 2.0

def testIsAbove():
    remover = initRemover()

    x_range = np.arange(0, 11, 5)
    y_range = np.arange(0, 11, 5)

    x, y = np.meshgrid(x_range, y_range, sparse=True)

    p1 = np.array([1, 1])
    p2 = np.array([10, 6])

    is_above = remover.isAbove(p1, p2, x, y)

    ideal_answer = np.array([[False, False, False],
                             [True,  True, False],
                             [True,  True,  True]])
    
    assert np.array_equal(is_above, ideal_answer)

def testIsOnRight():
    remover = initRemover()

    x_range = np.arange(0, 11, 5)
    y_range = np.arange(0, 11, 5)

    x, y = np.meshgrid(x_range, y_range, sparse=True)

    p1 = np.array([1, 1])
    p2 = np.array([10, 6])

    is_on_right = remover.isOnRight(p1, p2, x, y)

    ideal_answer = np.array([[True, True, True],
                             [False,  False, True],
                             [False,  False,  False]])

    assert np.array_equal(is_on_right, ideal_answer)

def testGenerateTextMask():
    remover = initRemover()

    img_height = 10
    img_width = 10

    vertices = np.array([[[0.5, 2.5],
                          [0.6, 0.6],
                          [2.5, 0.5],
                          [2.6, 2.6]]])    

    mask = remover.generateTextMask(vertices, img_height, img_width)
    
    ideal_mask = np.zeros((10, 10), dtype=np.uint8)
    ideal_mask[1:3, 1:3] = 255

    assert np.array_equal(mask, ideal_mask)

def testGenerateTextMaskWithMaskEnlargement():
    remover = initRemover()

    img_height = 10
    img_width = 10

    vertices = np.array([[[0.5, 2.5],
                          [0.6, 0.6],
                          [2.5, 0.5],
                          [2.6, 2.6]]])    

    mask = remover.generateTextMask(vertices, img_height, img_width, dilation=0.2)
    
    ideal_mask = np.zeros((10, 10), dtype=np.uint8)
    ideal_mask[:5, :5] = 255

    assert np.array_equal(mask, ideal_mask)


def testGetTextBoxes():
    img_array = np.array(img)
    remover = initRemover()

    text_boxes = remover.getTextBoxes(img_array)

    assert text_boxes.shape[1] == 4
    assert text_boxes.shape[2] == 2

    assert np.all(text_boxes[:,0,1] > text_boxes[:,1,1])
    assert np.all(text_boxes[:,2,0] > text_boxes[:,1,0])
    assert np.all(text_boxes[:,3,1] > text_boxes[:,2,1])
    assert np.all(text_boxes[:,3,0] > text_boxes[:,0,0])

def testInpaint():
    remover = initRemover()
    img_inpainted = remover.inpaint(img)

    assert isinstance(img_inpainted, Image.Image)

def initRemover():
    return SmartTextRemover()

if __name__ == "__main__":
    testCanInitSmartTextRemover()
    testConvertRgbToBgr()
    testFindGradientAndYIntersec()
    testIsAbove()
    testIsOnRight()
    testGenerateTextMask()
    testGenerateTextMaskWithMaskEnlargement()
    testGetTextBoxes()
    testInpaint()
