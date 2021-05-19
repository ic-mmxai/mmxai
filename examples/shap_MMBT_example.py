from mmxai.interpretability.classification.shap._explainer import Explainer
from mmxai.interpretability.classification.shap import _utils as utils
from mmf.models.mmbt import MMBT

import torch

def main():
    """ Example for how to use this explainer
    """
    # read data to try
    data_path = input("Enter input path to your data folder:")
    labels = utils.read_labels(data_path + "/train.jsonl", True)
    ids = [5643, 6937]
    target_labels = [l for l in labels if l['id'] in ids]
    # print(f"{target_labels = }")
    target_images, target_texts = utils.parse_labels(
        target_labels, img_to_array=True, separate_outputs=True)

    # model to explain
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    # Move model to GPU if cuda is available
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Explainer hyper params
    max_evals = 2
    batch_size = 1

    # test default partition algo
    explainer = Explainer(model, max_evals=max_evals, batch_size=batch_size)
    # text_shap_values = explainer.explain(
    #     target_images, target_texts, "fix_image")
    image_shap_values = explainer.explain(
        target_images, target_texts, "fix_text")
    # img_values, txt_values = explainer.explain(
    #     target_images, target_texts, mode="multimodal")

    # plots
    explainer.image_plot(image_shap_values)


if __name__ == "__main__":
    main()
