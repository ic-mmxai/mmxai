from mmxai.interpretability.classification.lime.object_detection import *
import os, json, cv2, random
from PIL import Image
import numpy as np
predictor,cfg = object_detection_predictor();
im1 = cv2.imread("98564.png")
im2 = cv2.imread("01236.png")

def test_initialization():
    try:
        predictor,cfg = object_detection_predictor();
    except:
        assert False, "Cannot initialize the predictor"
    else:
        assert True

def test_label():
    label = object_detection_obtain_label(predictor,cfg,im2)
    assert type(label) == np.ndarray
    assert type(label[0]) == np.str_

def test_compare():
    label1 = object_detection_obtain_label(predictor,cfg,im1)
    label2 = object_detection_obtain_label(predictor,cfg,im2)
    compared_label = compare_labels(label1,label2)
    assert type(compared_label) == np.ndarray
    assert type(compared_label[0]) == np.float64
