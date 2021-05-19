from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import multi_extremal_perturbation
from torchray.attribution.extremal_perturbation import contrastive_reward

import torch
import matplotlib.pyplot as plt

from mmf.models.mmbt import MMBT
from mmxai.interpretability.classification.torchray.extremal_perturbation.custom_mmbt import MMBTGridHMInterfaceOnlyImage

def testMultiExtremalPerturbationStandardCase():

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    text = "How I want to say hello to Asian people"

    model = MMBTGridHMInterfaceOnlyImage(
        MMBT.from_pretrained("mmbt.hateful_memes.images"), text)
    model = model.to(device)

    image_path = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"
    image_tensor = model.imageToTensor(image_path)

    # if device has some error just comment it
    image_tensor = image_tensor.to(device)

    _out, out, = multi_extremal_perturbation(model,
                                             torch.unsqueeze(image_tensor, 0),
                                             image_path,
                                             text,
                                             0,
                                             reward_func=contrastive_reward,
                                             debug=True,
                                             max_iter=200,
                                             areas=[0.12],
                                             show_text_result=True)

def testMultiExtremalPerturbationWithFloatMaskArea():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    text = "How I want to say hello to Asian people"

    model = MMBTGridHMInterfaceOnlyImage(
        MMBT.from_pretrained("mmbt.hateful_memes.images"), text)
    model = model.to(device)

    image_path = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"
    image_tensor = model.imageToTensor(image_path)

    # if device has some error just comment it
    image_tensor = image_tensor.to(device)

    _out, out, = multi_extremal_perturbation(model,
                                             torch.unsqueeze(image_tensor, 0),
                                             image_path,
                                             text,
                                             0,
                                             reward_func=contrastive_reward,
                                             debug=True,
                                             max_iter=200,
                                             areas=0.12,
                                             show_text_result=True)

def testMultiExtremalPerturbationWithDeleteVarient():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    text = "How I want to say hello to Asian people"

    model = MMBTGridHMInterfaceOnlyImage(
        MMBT.from_pretrained("mmbt.hateful_memes.images"), text)
    model = model.to(device)

    image_path = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"
    image_tensor = model.imageToTensor(image_path)

    # if device has some error just comment it
    image_tensor = image_tensor.to(device)

    _out, out, = multi_extremal_perturbation(model,
                                             torch.unsqueeze(image_tensor, 0),
                                             image_path,
                                             text,
                                             0,
                                             reward_func=contrastive_reward,
                                             debug=True,
                                             max_iter=200,
                                             areas=0.12,
                                             variant="delete",
                                             show_text_result=True)


def testMultiExtremalPerturbationWithSmoothMask():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    text = "How I want to say hello to Asian people"

    model = MMBTGridHMInterfaceOnlyImage(
        MMBT.from_pretrained("mmbt.hateful_memes.images"), text)
    model = model.to(device)

    image_path = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"
    image_tensor = model.imageToTensor(image_path)

    # if device has some error just comment it
    image_tensor = image_tensor.to(device)

    _out, out, = multi_extremal_perturbation(model,
                                             torch.unsqueeze(image_tensor, 0),
                                             image_path,
                                             text,
                                             0,
                                             reward_func=contrastive_reward,
                                             debug=True,
                                             max_iter=200,
                                             areas=[0.12],
                                             smooth=0.5,
                                             show_text_result=True)