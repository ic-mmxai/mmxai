from mmxai.interpretability.classification.shap import Explainer
import numpy as np
from PIL import Image
import os


def shap_multimodal_explain(
    image_name,
    text,
    model,
    label_to_exp,
    model_output,
    **kwargs
):
    """Interface for webapp to use shap explainer for multimodal classification

    Args:
        kwargs: arguments for Explainer see mmxai.interpretability.classification.shap

    Returns:
        what is returned:

    Raises:
        KeyError: An example
    """

    # parse and prepare input
    image = "static/" + image_name
    target_images = Image.open(image)
    target_images = np.array(target_images, dtype=np.uint8)
    if target_images.shape[2] > 3:
        target_images = target_images[:, :, :3]
    target_images = target_images[np.newaxis, ...]
    target_texts = np.array([text])

    # explain using the parameters given
    explainer = Explainer(model, **kwargs)
    image_shap_values, text_shap_values  = explainer.explain(
        target_images, target_texts, "multimodal"
    )

    # prepare image output
    PIL_image = explainer.image_plot(image_shap_values, label_index=label_to_exp)[0]
    name_split_list = os.path.splitext(image_name)
    exp_image = name_split_list[0] + "_shap_img.png"
    PIL_image.save("static/" + exp_image)

    # prepare text output
    text_exp = explainer.parse_text_values(text_shap_values, label_index=label_to_exp)
    text_exp = {
        k: v
        for k, v in sorted(text_exp[0].items(), key=lambda item: item[1], reverse=True)
    }

    # make explanation of outputs
    text_str, image_str = conclusion(
        image_shap_values[0],
        text_shap_values[0],
        model_output=model_output,
        label_index=label_to_exp,
    )

    return (text_exp, exp_image, text_str, image_str)


def conclusion(image_shap_values, text_shap_values, model_output, label_index):
    """Output text explanation for the explanation of shap

    Args:
        shap_values: shap values that contain the base values needed
            shape (..., #labels);
            ... corresponds to the input dimensions
            #labels = number of output labels

        model_output: array of shape (#labels, )
        label_index: the label that

    Returns:
        a string of explanation for web hover
    """
    assert (
        label_index < image_shap_values.base_values.shape[-1]
        and label_index < text_shap_values.base_values.shape[-1]
    ), f"label index {label_index} is not within the range of model outputs"

    def sum_over_all_but_last(arr):
        return arr.reshape(-1, arr.shape[-1]).sum(axis=0)

    img_sum = sum_over_all_but_last(image_shap_values.values)[label_index]
    txt_sum = sum_over_all_but_last(text_shap_values.values)[label_index]

    base_value = text_shap_values.base_values[label_index]

    num_img_features = np.unique(image_shap_values.segment_map).shape[0]
    num_txt_features = text_shap_values.data.shape[0]
    # num_unique_words = np.unique(text_shap_values.data).shape[0]

    def hateful(is_hateful):
        return "Hateful" if is_hateful else "Not Hateful"

    h = label_index == 1

    tldr = f'<p><strong>tl;dr:</strong><br> <span style="color: red">Red</span> (<span style="color: blue">Blue</span>) regions move the model output towards {hateful(h)} ({hateful(not h)}).</p>'
    img_details = (
        f"<p>"
        f"<strong>Details</strong>:<br>The input image is segmented into {num_img_features} regions, and text string is split into {num_txt_features} features. "
        f"The shapley values of those {num_img_features + num_txt_features} features represent their additive contributions towards the model output for the current inclination selected, {model_output[label_index]: .3f}, on top of the base value. The base value, {base_value:.4f}, is the expected model output without those features. "
        f"The sum of all shapley values and the base value should equate the selected model output, i.e."
        f"</p>"
        f"<p><em>model_output = base_value + total_image_shapley_values + total_text_shapley_values</em>.</p>"
        f"<p>The sum of shapley values for the image features is {img_sum:.4f}.</p>"
    )
    caveat = f'<span style="font-size:0.75rem">*note: the results may change slightly if the number of evaluations is small due to random sampling in the algorithm</span>'
    # add assertive msg
    assertive = (
        f"<p>Indeed, {model_output[label_index]: .3f} &#8776 {base_value:.4f} + {img_sum:.4f} + {txt_sum: .4f}.</p>"
        if (model_output[label_index] - (base_value + img_sum + txt_sum) < 1e-3)
        else ""
    )
    txt_details = f"<p><strong>Details</strong>:<br>The sum of shapley values of the {num_txt_features} text features is {txt_sum:.4f}.</p>"
    caveat2 = f'<span style="font-size:0.75rem">*note: if there are repeated words in the input, their shapley values are summed.</span>'

    return tldr + txt_details + assertive + caveat2, tldr + img_details + caveat
