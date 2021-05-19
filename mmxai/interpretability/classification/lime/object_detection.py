"""
Functions for object detection with detectron2
One extra part for extended LIME
"""
import numpy as np
import os, json, cv2, random
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,_create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog

#Configuration method to obtain config files and set parameters and checkpoint appropriately
def object_detection_predictor():
    """"
    Arguments:
        no input
    Returns:
        predictor: object predictor implementing COCO-Detection faster-rcnn backbone architecture
        cfg: object including parameters for the model like weights and threshold
    """
    cfg = get_cfg()
    # config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # config_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    config_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    #cfg.MODEL.WEIGHTS = "./checkpoint.pkl"  ########### the path to the checkpoint
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
    predictor = DefaultPredictor(cfg)  #
    return predictor, cfg

#Method to obtain detected labels in the input image
def object_detection_obtain_label(predictor, cfg, img):
    """"
    Arguments:
        predictor: object predictor implementing COCO-Detection faster-rcnn backbone architecture
        cfg: object including parameters for the model like weights and threshold
        img: image numpy array
    Returns:
        label: One numpy array containing only detected object names in the image(string)
    """
    outputs = predictor(img)
    predictions = outputs["instances"]
    scores = predictions.scores if predictions.has("scores") else None
    classes = (
        predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    )
    labels = _create_text_labels(
        classes,
        None,
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None),
    )
    label = np.unique(np.array(labels))
    return label

#Method to compare labels detected from original images and perturbed images
def compare_labels(o_label, n_label):
    """"
    Arguments:
        o_label: The label name numpy array detected from the original image
        o_label: The label name numpy array detected from the perturbed image
    Returns:
        n_label_np: A new numpy array with the same size of o_label. It is in one-hot
        representation where 1 represents the perturbed image can be detected one same
        label with the original one and 0 in opposite.
    """
    o_label_num = o_label.shape[0]
    n_label_num = n_label.shape[0]
    n_label_np = np.zeros(o_label_num)
    for i in range(n_label_num):
        for j in range(o_label_num):
            if n_label[i] == o_label[j]:
                n_label_np[j] = 1

    return n_label_np
