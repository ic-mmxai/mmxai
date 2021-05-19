from torchray.attribution.extremal_perturbation import contrastive_reward
from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import *
import torch
import matplotlib.pyplot as plt
import pytest
from mmf.models.mmbt import MMBT


Model = MMBT.from_pretrained("mmbt.hateful_memes.images")



def test_input_error():
    image_path = "hateful_asian.jpg"
    image_tensor = image2tensor(image_path)
    text = "How I want to say hello to Asian people"
    model = Model

    with pytest.raises(ValueError) as e:
        mask_, hist, x, summary, conclusion = multi_extremal_perturbation(model,
                                            torch.squeeze(image_tensor,0),
                                            image_path,
                                            text,
                                            0,
                                            reward_func=contrastive_reward,
                                            debug=True,
                                            max_iter=1,
                                            areas=[0.12]
                                            )
    assert str(e.value) == "Image tensor is suppose to be 4 dimensional"
    print(f"{str(e.value)=}")

    with pytest.raises(ValueError) as e:
        mask_, hist, x, summary, conclusion = multi_extremal_perturbation(model,
                                            image_tensor,
                                            image_path,
                                            "",
                                            0,
                                            reward_func=contrastive_reward,
                                            debug=True,
                                            max_iter=1,
                                            areas=[0.12]
                                            )
    assert str(e.value) == "Empty text"
    print(f"{str(e.value)=}")


    # test if error will arise when the input mask datatype is wrong
    with pytest.raises(ValueError) as e:
        mask_, hist, x, summary, conclusion = multi_extremal_perturbation(model,
                                            image_tensor,
                                            image_path,
                                            text,
                                            0,
                                            reward_func=contrastive_reward,
                                            debug=True,
                                            max_iter=1,
                                            areas=1.2
                                            )
    assert str(e.value) == "Mask must be a list contains a float value between 0 to 1"
    print(f"{str(e.value)=}")

    # test if error will arise when the non-standard input model is used
    with pytest.raises(ValueError) as e:
        mask_, hist, x, summary, conclusion = multi_extremal_perturbation(None,
                                            image_tensor,
                                            image_path,
                                            text,
                                            0,
                                            reward_func=contrastive_reward,
                                            debug=True,
                                            max_iter=1,
                                            areas=[0.12]
                                            )
    assert str(e.value) == "Model object must have a .classify attribute"
    print(f"{str(e.value)=}")

    # test if error will arise when the unsupported target is used
    with pytest.raises(ValueError) as e:
        mask_, hist, x, summary, conclusion = multi_extremal_perturbation(model,
                                            image_tensor,
                                            image_path,
                                            text,
                                            2,
                                            reward_func=contrastive_reward,
                                            debug=True,
                                            max_iter=1,
                                            areas=[0.12]
                                            )
    assert str(e.value) == "Target can be either 0 or 1"
    print(f"{str(e.value)=}")


def test_text_explainer_errors():
    image_path = "hateful_asian.jpg"
    image_tensor = image2tensor(image_path)
    text = "How I want to say hello to Asian people"
    model = Model

    Result = explain_text(text,torch.squeeze(image_tensor,0),model)


    word_list = text.split()
    

    summary, conclusion = Conclusion(text, Result)


    assert (len(word_list) == len(Result) == len(conclusion))



def test_multiModal_explainer_errors():
    image_path = "hateful_asian.jpg"
    image_tensor = image2tensor(image_path)
    text = "How I want to say hello to Asian people"
    model = Model

    mask_, hist, x, summary, conclusion = multi_extremal_perturbation(model,
                                             image_tensor,
                                             image_path,
                                             text,
                                             0,
                                             reward_func=contrastive_reward,
                                             debug=True,
                                             max_iter=1,
                                             areas=[0.12]
                                             )
    assert(len(x.shape) == 4)
    assert(x.shape == image_tensor.shape)

