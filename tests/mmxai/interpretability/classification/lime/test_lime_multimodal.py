from PIL import Image
from mmxai.interpretability.classification.lime.lime_multimodal import *
from mmf.models.mmbt import MMBT
from mmf.models.visual_bert import VisualBERT


# prepare image, text and model for the explanation generation pipeline
img_path = "tests/mmxai/interpretability/classification/lime/gun.jpeg"
img_try = Image.open(img_path)
text = "How I want to say hello to deliberately hateful Asian people, I hate them"
image_numpy = np.array(img_try)

model_mmbt = MMBT.from_pretrained("mmbt.hateful_memes.images")
model_visualbert = VisualBERT.from_pretrained(
                    "visual_bert.finetuned.hateful_memes.from_coco"
                )


# prediction using mock classification model object
def classifier_fn(model, imgs, txts, zero_image=False, zero_text=False):
    inputs = zip(imgs, txts)
    res = np.zeros((len(imgs), 2))
    for i, this_input in enumerate(inputs):
        img = Image.fromarray(this_input[0])
        txt = this_input[1]
        try:
            this_output = model.classify(
                img, txt, zero_image=zero_image, zero_text=zero_text
            )
        except:
            this_output = model.classify(img, txt)
        res[i][this_output["label"]] = this_output["confidence"]
        res[i][1 - this_output["label"]] = 1 - this_output["confidence"]
    return res


def test_explainer_init():
    try:
        exp1 = LimeMultimodalExplainer(image_numpy, text, model_mmbt)
    except:
        assert False, "Cannot initialise"
    else:
        assert True


# used multiple times, instantiate a global object to reduce computation
exp1 = LimeMultimodalExplainer(image_numpy, text, model_mmbt)
exp2 = LimeMultimodalExplainer(image_numpy, text, model_visualbert)
num_sample = 100
num_sample_2 = 2


def test_data_labels():
    # use number of samples value % 10 != 0
    # get data, labels, distances to fit the linear model
    (
        data,
        labels,
        distances,
        n_txt_features,
        n_img_features,
        segments,
        domain_mapper,
        n_detection_features,
        detection_label,
        ratio_txt_img,
    ) = exp1.data_labels(num_sample, classifier_fn)    
    n_features = n_img_features + n_txt_features
    assert data.shape == (num_sample + 1, n_features)
    assert labels.shape == (num_sample + 1, 2)
    assert segments.shape
    assert domain_mapper
    assert distances.shape == (num_sample + 1,)
    assert n_detection_features == 0
    assert detection_label == None
    assert ratio_txt_img 

    # use number of samples value % 10 == 0
    (
        data,
        labels,
        distances,
        n_txt_features,
        n_img_features,
        segments,
        domain_mapper,
        n_detection_features,
        detection_label,
        ratio_txt_img,
    ) = exp1.data_labels(num_sample_2, classifier_fn)   
    n_features = n_img_features + n_txt_features
    assert data.shape == (num_sample_2 + 1, n_features)
    assert labels.shape == (num_sample_2 + 1, 2)
    assert segments.shape
    assert domain_mapper
    assert distances.shape == (num_sample_2 + 1,)

    # Don't use zero_text and image
    (
        data,
        labels,
        distances,
        n_txt_features,
        n_img_features,
        segments,
        domain_mapper,
        n_detection_features,
        detection_label,
        ratio_txt_img,
    ) = exp2.data_labels(num_sample_2, classifier_fn)   
    n_features = n_img_features + n_txt_features


# used multiple times, generate a global object
(
    data,
    labels,
    distances,
    n_txt_features,
    n_img_features,
    segments,
    domain_mapper,
    n_detection_features,
    detection_label,
    ratio_txt_img,
) = exp1.data_labels(num_sample, classifier_fn)    
n_features = n_img_features + n_txt_features


def test_explain_instance():
    explanation1 = exp1.explain_instance(classifier_fn, 50)
    assert explanation1

    unsorted_weights = list(explanation1.unsorted_weights[0])
    assert len(unsorted_weights) == n_features

    sorted_weights = list(explanation1.local_exp[0])
    assert len(sorted_weights) == n_features

    assert 0 <= explanation1.score[0] <= 1


# global object used in later tests
explanation1 = exp1.explain_instance(classifier_fn, 50)


def test_explanation_image_case():

    # case1, positive=False, negative=False
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    assert img.shape == image_numpy.shape
    assert mask.shape

    # case2, positive=True, negative=False
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=True,
                                                num_features=10,
                                                hide_rest=True)
    assert img.shape == image_numpy.shape
    assert mask.shape

    # case3, positive=False, negative=True
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=False,
                                                negative_only=True,
                                                num_features=10,
                                                hide_rest=True)
    assert img.shape == image_numpy.shape
    assert mask.shape

    # case4, label not exists
    try:
        img, mask = explanation1.get_image_and_mask(3,
                                                positive_only=False,
                                                negative_only=True,
                                                num_features=10,
                                                hide_rest=True)
    except KeyError as e:
        assert "not in explanation" in str(e)

    # case5, positive = negative = True
    try:
        img, mask = explanation1.get_image_and_mask(0,
                                                positive_only=True,
                                                negative_only=True,
                                                num_features=10,
                                                hide_rest=True)
    except ValueError as e:
        assert "cannot be true at the same time" in str(e)


def test_explanation_text():
    ans = explanation1.as_list()
    assert len(ans) == n_txt_features


def test_get_explanation():
    message = explanation1.get_txt_img_ratio()
    assert len(message) > 0

    txt_message, img_message, text_list, temp, mask = explanation1.get_explanation(0)
    assert len(txt_message) > 0
    assert len(img_message) > 0
    assert len(text_list) > 0
    assert temp.shape == image_numpy.shape
    assert mask.shape
    txt_message, img_message, text_list, temp, mask = explanation1.get_explanation(0, which_exp="negative")
    # excessive amount of features
    txt_message, img_message, text_list, temp, mask = explanation1.get_explanation(0, num_features=10000)


# coverage run -m pytest tests/mmxai/interpretability/classification/lime/test_lime_multimodal.py 
# coverage report -m mmxai/interpretability/classification/lime/lime_multimodal.py
