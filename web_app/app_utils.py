from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import re
from mmxai.onnx.onnxModel import ONNXInterface


class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def message(self):
        return self.msg


# raise errors(message) in prepare_explanation, catch in app.py calls, flash message
def prepare_explanation(
    img_name, img_text, user_model, model_type, model_path, encourage
):
    if model_path is not None:
        model_path = "static/" + model_path
    img_name = "static/" + img_name

    try:
        model = setup_model(user_model, model_type, model_path)
    except InputError as e:
        raise InputError(e.message()) from e

    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    img = Image.open(img_name)

    try:
        output = model.classify(img, img_text)
    except:
        raise InputError("The machine learning model cannot process the provided input")

    if encourage == "encourage":
        label_to_explain = output["label"]
    else:
        label_to_explain = abs(output["label"] - 1)
    label = output["label"]
    conf = output["confidence"]

    return model, label_to_explain, label, conf


# get model output in binary classification format with 2 labels """
def model_output(cls_label, cls_confidence):
    out = np.zeros(2)
    out[cls_label] = cls_confidence
    out[1 - cls_label] = 1 - cls_confidence
    return out


def setup_model(user_model, model_type, model_path):
    if user_model == "no_model":
        try:
            if model_type == "MMBT":
                model = MMBT.from_pretrained("mmbt.hateful_memes.images")
            elif model_type == "LateFusion":
                model = LateFusion.from_pretrained("late_fusion.hateful_memes")
            elif model_type == "ViLBERT":
                model = ViLBERT.from_pretrained(
                    "vilbert.finetuned.hateful_memes.from_cc_original"
                )
            else:  # visual bert
                model = VisualBERT.from_pretrained(
                    "visual_bert.finetuned.hateful_memes.from_coco"
                )
        except:
            raise InputError(
                "Sorry, having trouble opening the models we provided, please try again later."
            )

    elif user_model == "mmf":
        try:
            if model_type == "MMBT":
                model = MMBT.from_pretrained(model_path)
            elif model_type == "LateFusion":
                model = LateFusion.from_pretrained(model_path)
            elif model_type == "ViLBERT":
                model = ViLBERT.from_pretrained(model_path)
            else:
                model = VisualBERT.from_pretrained(model_path)
        except:
            raise InputError(
                "Sorry, we cannot open the mmf checkpoint you uploaded. It should be an .ckpt file saved from the mmf trainer."
            )

    elif user_model == "onnx":
        model = ONNXInterface(model_path, model_type)

    else:
        raise InputError("Please select a model upload type")

    return model


def check_image(image_name):
    try:
        img = Image.open(image_name)
        img = np.array(img, dtype=np.uint8)
        # If the input image has 4 channel, discard the alpha channel
        if img.shape[2] > 3:
            img = img[:, :, :3]
            img = Image.fromarray(img)
            img.save(image_name)
    except:
        raise InputError(
            "Sorry, we cannot open your uploaded image. Please check the file format."
        )


def check_text(image_txt):
    try:
        for i in image_txt:
            assert i.encode( 'UTF-8' ).isalnum() or i == " " or i == "," or i == "." or i == "?" or i == "!" or i == "'"
        image_txt = re.sub('[^A-Za-z0-9 ,.!?]+', "", image_txt)
        splitted = image_txt.split()
        res = "".join((i + " ") for i in splitted).strip()
        assert len(res) > 0
        return res
    except:
        raise InputError(
            "For the text inputs, please use English alphabets, numbers, common punctuations and blank spaces only."
        )


def text_visualisation(exp, pred_res, save_path):
    if pred_res == 1:
        plt_title = "hateful"
    else:
        plt_title = "not hateful"

    # handle different output formats from explainers
    vals = []
    names = []
    if isinstance(exp, dict):
        for i in exp:
            names.append(i)
            vals.append(exp[i])
    elif isinstance(exp, list) and isinstance(exp[0], str):
        for i in exp:
            names.append(i.split()[0])
            vals.append(float(i.split()[-1]))
    elif isinstance(exp, list) and isinstance(exp[0], list):
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]

    vals.reverse()
    names.reverse()

    fig = plt.figure(figsize=(15, 0.75 + 0.375 * len(vals)))

    colors = ["hotpink" if x > 0 else "cornflowerblue" for x in vals]
    pos = np.arange(len(exp)) + 0.5
    plt.barh(pos, vals, align="center", color=colors)

    plt.yticks(pos, names, fontsize=25)
    plt.xticks(fontsize=15)

    plt.savefig(
        "static/" + save_path, bbox_inches="tight", transparent=True, pad_inches=0.1
    )


def read_examples_metadata(path="static/examples/metadata.json"):
    with open(path) as f:
        s = json.load(f)
    return s


def random_select(arr, **kwargs):
    return np.random.choice(arr, **kwargs)




