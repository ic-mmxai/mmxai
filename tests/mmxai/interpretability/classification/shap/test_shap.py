"""
===============================================================================
test_shap.py
test file for mmxai.interpretability.classification.shap
===============================================================================
"""
from mmxai.interpretability.classification.shap import ShapExplainer as Explainer
from mmxai.interpretability.classification.shap import utils
from mmf.models.mmbt import MMBT
import numpy as np
import PIL.Image as Image
import torch
import pytest

# the path for train.jsonl file for the hateful memes data
DATA_PATH = "tests/mmxai/interpretability/classification/shap/train.jsonl"
# =============================== fixtures =====================================
# hyperparams for fast run
@pytest.fixture(scope="module")
def global_data():
    max_evals = 2
    batch_size = 1
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    # Workaround for adding __call__ method to MMBT model
    def call(self, images: np.ndarray, texts: np.ndarray):
        out = np.zeros((images.shape[0], 2))
        for i, (text, image) in enumerate(zip(texts, images)):
            image = Image.fromarray(image)
            ind, score = self.classify(image, text).values()
            out[i][ind] = score
            out[i][1 - ind] = 1 - score
        return out
    model.__class__.__call__ = call

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = utils.read_labels(DATA_PATH, True)
    ids = [5643] # single input
    # ids = [5643, 6937] # multiple inputs - tested
    target_labels = [l for l in labels if l["id"] in ids]
    target_images, target_texts = utils.parse_labels(
        target_labels, img_to_array=True, separate_outputs=True
    )
    # reshape the images to the same shape (100, 100, 3)
    target_images = np.vstack([im[np.newaxis, :100, :100, ...] for im in target_images])

    outputs = model_outputs(model, target_images, target_texts)

    return {
        "max_evals": max_evals,
        "batch_size": batch_size,
        "model": model,
        "target_images": target_images,
        "target_texts": target_texts,
        "outputs": outputs,
    }


# ============================= helper functions ===============================


def sum_over_all_but_last(arr):
    return arr.reshape(-1, arr.shape[-1]).sum(axis=0)


def model_outputs(model, images, texts):
    out = np.zeros((len(texts), 2))  # output same shape
    images = utils.arr_to_img(images)
    for i, (text, image) in enumerate(zip(texts, images)):
        # classify, output is a tupe (index, score)
        ind, score = model.classify(image, text).values()
        out[i][ind] = score
        out[i][1 - ind] = 1 - score
    return out


# ============================== tests ========================================

# test utilities above
def test_utils_read_data():
    labels = utils.read_labels(DATA_PATH, True)
    ids = [5643, 6937]
    target_labels = [l for l in labels if l["id"] in ids]
    target_images, target_texts = utils.parse_labels(
        target_labels, img_to_array=True, separate_outputs=True
    )
    assert len(target_labels) == len(ids)
    assert target_images.shape[0] == len(ids)
    assert target_texts.shape[0] == len(ids)


# test initialisation validations
def test_explainer_init(global_data):
    try:
        explainer = Explainer(
            global_data["model"],
            max_evals=global_data["max_evals"],
            batch_size=global_data["batch_size"],
        )
    except:
        assert False, "Failed to initiate the explainer"
    else:
        assert True

    # test whether the ValueError will be raised if the input algorithm is not supported
    with pytest.raises(ValueError) as e:
        explainer = Explainer(
            global_data["model"],
            algorithm="other",
            max_evals=global_data["max_evals"],
            batch_size=global_data["batch_size"],
        )
    assert "is not supported" in str(e.value)
    print(f"{str(e.value)=}")

    # test whether the ValueError will be raised if the input model is not correct
    with pytest.raises(ValueError) as e:
        explainer = Explainer(
            None,
            max_evals=global_data["max_evals"],
            batch_size=global_data["batch_size"],
        )
    assert "callable" in str(e.value) 
    print(f"{str(e.value)=}")

# test input validations
def test_explain_errors(global_data):
    explainer = Explainer(
        global_data["model"],
        algorithm="partition",
        max_evals=global_data["max_evals"],
        batch_size=global_data["batch_size"],
    )
    with pytest.raises(ValueError) as e:
        text_shap_values = explainer.explain(
            global_data["target_images"], global_data["target_texts"], "another_mode"
        )
    assert "mode" in str(e) and "not supported" in str(e.value)
    print(f"{str(e.value)=}")


# test explain() method
def test_explain_partition(global_data):
    explainer = Explainer(
        global_data["model"],
        algorithm="partition",
        max_evals=global_data["max_evals"],
        batch_size=global_data["batch_size"],
    )

    img_values, txt_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], mode="multimodal"
    )
    # should have used the class in utils as tokenizer
    assert isinstance(explainer.tokenizer, utils.WhitespaceTokenizer)

    text_shap_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], "multimodal_fix_image"
    )
    image_shap_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], "multimodal_fix_text"
    )

    # assert length and shapes
    assert (
        len(text_shap_values)
        == len(txt_values)
        == len(image_shap_values)
        == len(img_values)
        == len(global_data["outputs"])
    )
    assert all([t1.shape == t2.shape for t1, t2 in zip(image_shap_values, img_values)])

    # check all the shap values add up to model output - correctness of the method
    for i, out in enumerate(global_data["outputs"]):
        s_txt = (
            sum_over_all_but_last(text_shap_values[i].values)
            + text_shap_values[i].base_values
        )
        s_img = (
            sum_over_all_but_last(image_shap_values[i].values)
            + image_shap_values[i].base_values
        )
        s_mm = (
            sum_over_all_but_last(img_values[i].values)
            + sum_over_all_but_last(txt_values[i].values)
            + txt_values[i].base_values
        )
        print(f"{s_txt=}")
        print(f"{s_img=}")
        print(f"{s_mm=}")
        assert np.isclose(out, s_txt, rtol=1e-04, atol=1e-07).all()
        assert np.isclose(out, s_img, rtol=1e-04, atol=1e-07).all()
        assert np.isclose(out, s_mm, rtol=1e-04, atol=1e-07).all()


# test explain() method
def test_explain_permutation(global_data):
    explainer = Explainer(
        global_data["model"],
        algorithm="permutation",
        max_evals=global_data["max_evals"],
        batch_size=global_data["batch_size"],
    )
    text_shap_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], "multimodal_fix_image"
    )
    with pytest.raises(ValueError) as e:
        image_shap_values = explainer.explain(
            global_data["target_images"], global_data["target_texts"], "multimodal_fix_text"
        )
    assert "it will take too long" in str(e.value)
    print(f"{str(e.value)=}")

    img_values, txt_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], mode="multimodal"
    )

    # assert length and shapes
    assert (
        len(text_shap_values)
        == len(txt_values)
        == len(img_values)
        == len(global_data["outputs"])
    )

    # check all the shap values add up to model output - correctness of the method
    for i, out in enumerate(global_data["outputs"]):
        s_txt = (
            sum_over_all_but_last(text_shap_values[i].values)
            + text_shap_values[i].base_values
        )
        s_mm = (
            sum_over_all_but_last(img_values[i].values)
            + sum_over_all_but_last(txt_values[i].values)
            + txt_values[i].base_values
        )
        print(f"{s_txt=}")
        print(f"{s_mm=}")
        assert np.isclose(out, s_txt, rtol=1e-04, atol=1e-07).all()
        assert np.isclose(out, s_mm, rtol=1e-04, atol=1e-07).all()


# test plotting
def test_image_plot(global_data):
    explainer = Explainer(
        global_data["model"],
        algorithm="partition",
        max_evals=global_data["max_evals"],
        batch_size=global_data["batch_size"],
    )
    image_shap_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], "multimodal_fix_text"
    )
    plots = explainer.image_plot(image_shap_values)
    # plots[0].show()
    assert len(plots) == len(global_data["target_images"])


# test parsing txt shap values():
def test_parse_text_values(global_data):
    explainer = Explainer(
        global_data["model"],
        algorithm="partition",
        max_evals=global_data["max_evals"],
        batch_size=global_data["batch_size"],
    )
    text_shap_values = explainer.explain(
        global_data["target_images"], global_data["target_texts"], "multimodal_fix_image"
    )
    txt_dics = explainer.parse_text_values(text_shap_values)
    # print(txt_dics)
    assert len(txt_dics) == len(global_data["target_texts"])

if __name__ == "__main__":
    test_utils_read_data()
