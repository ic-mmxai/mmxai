# TODO: could use mock objects to improve test efficiency

from mmxai.interpretability.classification.torchray.extremal_perturbation.custom_mmbt import MMBTGridHMInterfaceOnlyImage

from mmf.models.mmbt import MMBT

import torch
from PIL import Image
import requests


def testObjectInitiation():
    try:
        model = MMBTGridHMInterfaceOnlyImage(
            MMBT.from_pretrained("mmbt.hateful_memes.images"), "test text")
    except:
        assert False, "cannot instantiate MMBTGridHMInterfaceOnlyImage object"
    else:
        assert True


# instantiate a model globally for better testing efficiency
MODEL = MMBTGridHMInterfaceOnlyImage(
    MMBT.from_pretrained("mmbt.hateful_memes.images"), "test text")
MODEL = MODEL.to(torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"))


def testCanGetTextAttribute():
    text = MODEL.text

    assert text == "test text"


def testCanClassifyMultiModalInputs():
    image_path = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"
    text = "How I want to say hello to Asian people"
    try:
        MODEL.classify(image_path, text)
    except:
        assert False, "cannot classify multimodal inputs"
    else:
        assert True


def testImageToTensorFromUrl():
    url = "https://img.17qq.com/images/ghhngkfnkwy.jpeg"

    image_tensor = MODEL.imageToTensor(url)

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == torch.Size([3, 224, 224])


def testImageToTensorFromLocalPath():
    path = "tests/mmxai/interpretability/classification/torchray/extremal_perturbation/test.jpg"

    image_tensor = MODEL.imageToTensor(path)

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == torch.Size([3, 224, 224])


def testObjectIsCallableWithOnlyImageInput():
    score = MODEL(torch.rand(3, 224, 224))

    assert isinstance(score, torch.Tensor)
    assert score.shape == torch.Size([1, 2])


def testObjectIsCallableWithBothImageAndTextInputs():
    score = MODEL(torch.rand(3, 224, 224), "test text")

    assert isinstance(score, torch.Tensor)
    assert score.shape == torch.Size([1, 2])


if __name__ == "__main__":
    pass
