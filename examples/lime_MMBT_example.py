from mmf.models.mmbt import MMBT
from mmxai.interpretability.classification.lime.lime_multimodal import LimeMultimodalExplainer
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import torch
import json
import re


model = MMBT.from_pretrained("mmbt.hateful_memes.images")


def text_visualisation(exp, pred_res, img_path, method="lime"):
    if pred_res == 1:
        plt_title = "hateful"
    else:
        plt_title = "not hateful"

    print(exp)

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
    
    name_split_list = img_path.split(".")
    exp_txt = name_split_list[0] + f"_{method}_txt.png"
    print(exp_txt)
    
    plt.savefig(
        exp_txt, bbox_inches="tight", transparent=True, pad_inches=0.1
    )


def lime_multimodal_explain(image_path, text, model, label_to_exp, num_samples=1200):
    image = image_path
    img_try = Image.open(image)
    text = text
    image_numpy = np.array(img_try)

    def multi_predict(model, imgs, txts, zero_image=False, zero_text=False):
        inputs = zip(imgs, txts)
        res = np.zeros((len(imgs), 2))
        for i, this_input in enumerate(inputs):
            img = Image.fromarray(this_input[0])
            txt = this_input[1]
            this_output = model.classify(
                img, txt, zero_image=zero_image, zero_text=zero_text
            )
            res[i][this_output["label"]] = this_output["confidence"]
            res[i][1 - this_output["label"]] = 1 - this_output["confidence"]
        return res

    exp1 = LimeMultimodalExplainer(image_numpy, text, model)
    explanation1 = exp1.explain_instance(multi_predict, num_samples)
    txt_message, img_message, text_list, temp, mask = explanation1.get_explanation(
        label_to_exp
    )
    img_boundry = mark_boundaries(temp, mask)
    img_boundry = img_as_ubyte(img_boundry)
    PIL_image = Image.fromarray(np.uint8(img_boundry)).convert("RGB")

    name_split_list = image_path.split(".")
    exp_image = name_split_list[0] + "_lime_img." + name_split_list[1]
    print(exp_image)
    PIL_image.save(exp_image)

    text_exp_list = []
    for pair in text_list:
        text_exp_list.append(list(pair))
    
    def get_second(element):
        return element[1]
    txt_output = sorted(text_exp_list, key=get_second, reverse=True)
    text_visualisation(txt_output, label_to_exp, image_path, "lime")
    return (
        txt_output,
        exp_image,
        txt_message,
        img_message,
    )


img_name = "" # replace this with image path
img_text = "" # replace this with memes text

img_try = Image.open(img_name)

# predict
output = model.classify(img_try, img_text)
label_to_explain = output["label"]

plt.imshow(img_try)
plt.axis("off")
plt.show()
hateful = "Yes" if output["label"] == 1 else "No"
print("Hateful as per the model?", hateful)
print(f"Model's confidence: {output['confidence'] * 100:.3f}%")

# explain using lime
text_exp, img_exp, txt_msg, img_msg = lime_multimodal_explain(
                img_name,
                img_text,
                model,
                label_to_explain,
                num_samples=30000,
            )
print(txt_msg, "\n", img_msg)

