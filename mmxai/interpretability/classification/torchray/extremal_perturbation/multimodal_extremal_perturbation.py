from __future__ import division
from __future__ import print_function

import sklearn
from sklearn.utils import check_random_state
import scipy as sp
import copy


import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, lars_path

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchray.utils import imsmooth, imsc
from torchray.attribution.common import resize_saliency
from torchray.attribution.extremal_perturbation import (
    MaskGenerator,
    Perturbation,
    simple_reward,
    contrastive_reward,
)

from PIL import Image
from torchray.utils import imsc
from torchvision import transforms

BLUR_PERTURBATION = "blur"
"""Blur-type perturbation for :class:`Perturbation`."""

FADE_PERTURBATION = "fade"
"""Fade-type perturbation for :class:`Perturbation`."""

PRESERVE_VARIANT = "preserve"
"""Preservation game for :func:`extremal_perturbation`."""

DELETE_VARIANT = "delete"
"""Deletion game for :func:`extremal_perturbation`."""



def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(img), quiet=False)
    img = img.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return torch.reshape(img, (1, 3, 224, 224))

def PIL2tensor(image_PIL):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(image_PIL), quiet=False)
    return torch.reshape(img, (1, 3, 224, 224))






# --------------------------explainer of both image & text----------------------------




def multi_extremal_perturbation(
    model,
    input_img,
    input_text,
    target,
    areas=[0.1],
    perturbation=BLUR_PERTURBATION,
    max_iter=800,
    num_levels=8,
    step=7,
    sigma=21,
    jitter=True,
    variant=PRESERVE_VARIANT,
    print_iter=None,
    debug=False,
    reward_func=simple_reward,
    resize=False,
    resize_mode="bilinear",
    smooth=0,
    text_explanation_plot=False,
):
    r"""Compute a set of extremal perturbations.

    The function takes a :attr:`model`, an :attr:`input` tensor :math:`x`
    of size :math:`1\times C\times H\times W`, and a :attr:`target`
    activation channel. It produces as output a
    :math:`K\times C\times H\times W` tensor where :math:`K` is the number
    of specified :attr:`areas`.

    Each mask, which has approximately the specified area, is searched
    in order to maximise the (spatial average of the) activations
    in channel :attr:`target`. Alternative objectives can be specified
    via :attr:`reward_func`.

    Args:
        model (:class:`torch.nn.Module`): model.
        input (:class:`torch.Tensor`): input tensor.
        target (int): target channel.
        areas (float or list of floats, optional): list of target areas for saliency
            masks. Defaults to `[0.1]`.
        perturbation (str, optional): :ref:`ep_perturbations`.
        max_iter (int, optional): number of iterations for optimizing the masks.
        num_levels (int, optional): number of buckets with which to discretize
            and linearly interpolate the perturbation
            (see :class:`Perturbation`). Defaults to 8.
        step (int, optional): mask step (see :class:`MaskGenerator`).
            Defaults to 7.
        sigma (float, optional): mask smoothing (see :class:`MaskGenerator`).
            Defaults to 21.
        jitter (bool, optional): randomly flip the image horizontally at each iteration.
            Defaults to True.
        variant (str, optional): :ref:`ep_variants`. Defaults to
            :attr:`PRESERVE_VARIANT`.
        print_iter (int, optional): frequency with which to print losses.
            Defaults to None.
        debug (bool, optional): If True, generate debug plots.
        reward_func (function, optional): function that generates reward tensor
            to backpropagate.
        resize (bool, optional): If True, upsamples the masks the same size
            as :attr:`input`. It is also possible to specify a pair
            (width, height) for a different size. Defaults to False.
        resize_mode (str, optional): Upsampling method to use. Defaults to
            ``'bilinear'``.
        smooth (float, optional): Apply Gaussian smoothing to the masks after
            computing them. Defaults to 0.

    Returns:
        A tuple containing the masks and the energies.
        The masks are stored as a :class:`torch.Tensor`
        of dimension
    """

        
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input_img.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input_img.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(input_img, num_levels=num_levels, type=perturbation).to(
        device
    )

    perturbation_str = "\n  ".join(perturbation.__str__().split("\n"))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, : int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD(
        [pmask], lr=learning_rate, momentum=momentum, dampening=momentum
    )
    hist = torch.zeros((len(areas), 2, 0))



    if input_img == None:

        if input_text != None:
            text_Result = explain_text(input_text, None, model)
            if text_explanation_plot:
                explainer_plot(input_text, text_Result)
            summary, conclusion = Conclusion(input_text, text_Result)

        result["text_exp"] = summary
        result["high_level_exp"] = conclusion
        return result


    for t in range(max_iter):

        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)

        else:
            if_false = 1
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.

        if input_text == None:
            inputx = x.reshape(1,x.shape[1],x.shape[2],3)
            y = model(inputx,input_text)
        else:
            inputx = x.reshape(1,x.shape[1],x.shape[2],3)
            y = model(inputx)

        # update text
        if (t+1) % 400 == 0 and max_iter >= 800 and input_text != None and target <=1:
            Result = explain_text(input_text, torch.squeeze(x, 0), model)
            Not_hateful, Hateful = text_rebuilder(input_text, Result)
            temp_text = Hateful

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = -((mask_sorted - reference) ** 2).mean(dim=1) * regul_weight

        energy = (reward + regul).sum()
        optimizer.zero_grad()
        (-energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (
                hist,
                torch.cat(
                    (
                        reward.detach().cpu().view(-1, 1, 1),
                        regul.detach().cpu().view(-1, 1, 1),
                    ),
                    dim=1,
                ),
            ),
            dim=2,
        )

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input_img, mask_, resize, mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_, sigma=smooth * min(mask_.shape[2:]), padding_mode="constant"
        )

    if input_text != None:
        inputImage = input_img.reshape(1,input_img.shape[1],input_img.shape[2],3)
        text_Result = explain_text(input_text, inputImage, model)
        if text_explanation_plot:
            explainer_plot(input_text, text_Result)
        summary, conclusion = Conclusion(input_text, text_Result)
    else:
        summary = None
        conclusion = None

        
    # summary is the high level summary of the text explainer,
    # and the conclusion are the different words and corresponding scores
    result = dict()
    result["mask_tensor"] = mask_
    result["energy_during_training"] = hist
    result["image_exp"] = x
    result["text_exp"] = summary
    result["high_level_exp"] = conclusion
    return result


# --------------------------text explainer functions----------------------------------
def Conclusion(input_text, score):
    word_list = input_text.split()
    sorted_sequence = np.argsort(score)

    sorted_text = []
    conclusion = []
    for i, s in enumerate(sorted_sequence):
        sorted_text.append(word_list[s])
        conclusion.append(word_list[s] + " : " + str(score[s]))

    summary = "<p>The words : "
    for i in range(len(sorted_text)):
        i = len(sorted_text) - i - 1
        if score[sorted_sequence][i] > 0 and i > len(sorted_text) - 1 - 3:
            summary += "{" + sorted_text[i] + "}, "
    summary += 'are the most significant features that support Not hateful result.</p><p style="margin-bottom: -2px">The words : '

    for i in range(len(sorted_text)):

        if score[sorted_sequence][i] < 0 and i < 3:
            summary += "{" + sorted_text[i] + "}, "
    summary += "are the most significant features that support hateful result.</p>"
    return summary, conclusion


def distance_fn(x):
    """
    args :
        x : masked vectors to represent string

    returns :
        cosine distance between original and masked text
    """
    return (
        sklearn.metrics.pairwise.pairwise_distances(x, x[0], metric="cosine").ravel()
        * 100
    )


def masking(string_list, mask_indice):
    """
    args :
        string_list : list of individual words in the original text
        mask_indice : index of masked words

    returns :
        string of masked text
    """
    string_copy = copy.deepcopy(string_list)
    for i in mask_indice:
        string_copy[i] = "NOWORD"
    strout = "".join([i + " " for i in string_copy])
    return strout[:-1]


def text_explainer(string, img_tensor, classifier_fn):
    """
    args :
        string : original text
        img_tensor : tensor of input image
        classifier_fn : classification model

    returns :
        text_array : masked text vectors
        labels : predicted score from classifier
        distance : distance between masked text and original one
        (treat it as the weight of each text sample)
    """

    indexed = string.split()

    doc_size = len(indexed)
    num_samples = min(2 ** doc_size, 800)

    random_state = check_random_state(None)
    sample = random_state.randint(1, doc_size + 1, num_samples - 1)

    text_array = np.ones((num_samples, doc_size))

    features_range = range(doc_size)
    masked_text = [string]
    for i, size in enumerate(sample, start=1):
        mask = random_state.choice(features_range, size, replace=False)
        text_array[i, mask] = 0
        masked_text.append(masking(indexed, mask))
    distances = distance_fn(sp.sparse.csr_matrix(text_array))
    labels = []

    for i in range(len(masked_text)):
        if img_tensor != None:
            temp = (
                classifier_fn(img_tensor,masked_text[i])
                .detach()
                .cpu()
                .numpy()
            )
        else:
            temp = (
                classifier_fn(masked_text[i])
                .detach()
                .cpu()
                .numpy()
            )

        labels.append(temp[0])

    return text_array, np.array(labels), distances


def text_rebuilder(input_text, weight_factor):
    """
    args :
        input_text : original text
        weight_factor : the influence factor of different words


    returns :
        Not_hateful : New text with all the words that classified as hateful been masked
        Hateful New text with all the words that classified as not hateful been masked
    """
    Not_hateful = ""
    Hateful = ""
    txt = input_text.split()
    for i in range(len(weight_factor)):
        if weight_factor[i] < 0:
            Hateful += txt[i] + " "
            Not_hateful += "NOWARDS "
        else:
            Hateful += "NOWARDS "
            Not_hateful += txt[i] + " "
    return Not_hateful, Hateful


def explain_text(input_text, X, model):
    """
    args :
        input_text : original text
        X : image tensor (3,224,244)
        model : classification model that is going to be explained

    returns :
        Results : weights of the linear model to represent to influence of
                each words to the decision making
    """
    vectorized_text, labels, distances = text_explainer(input_text, X, model)

    weights = np.sqrt(np.exp(-(distances ** 2) / 25 ** 2))

    linear_model = Ridge(alpha=1, fit_intercept=True)
    linear_model.fit(vectorized_text, labels, sample_weight=weights)

    Result = linear_model.coef_[0]
    return Result


def explainer_plot(input_text, Result):
    """
    args :
        input_text : original text
        Results : weights of the linear model to represent to influence of
                each words to the decision making

    Visualize the result with bar chart
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    temp = input_text.split()
    for i in range(len(temp)):
        if temp[i] in temp[:i]:
            temp[i] += " "
    ax.barh(temp, Result, color=(0.4, 0.5, 0.8, 0.7))
    ax.axvline(0, color="red", alpha=1)
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.6)
    plt.xlabel("Relative influence towards Not Hateful Meme")
    plt.ylabel("Words")
    plt.text(max(Result) / 2, len(Result) - 1, "Not Hateful")
    plt.text(min(Result) / 2, len(Result) - 1, "Hateful")
    plt.show()




