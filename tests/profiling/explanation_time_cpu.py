import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

from app_utils import prepare_explanation, model_output
from interpretability4mmf import shap_mmf, lime_mmf, torchray_mmf
import cProfile

device = "cpu"

def predict(model_type,
            exp_method,
            user_model="no_model",
            img_name="profiling.png",
            img_text="kill the jews and muslims for being anti-white",
            model_path=None,
            exp_direction="encourage"):

    model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        img_name, img_text, user_model, model_type, model_path, exp_direction)

    hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'

    cls_result = 'Your uploaded image and text combination ' \
                 'looks like a {} meme, with {}% confidence.'.format(
                     hateful, "%.2f" % (cls_confidence * 100))

    print(cls_result)

    if exp_method == 'shap':
        text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
            img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
    elif exp_method == 'lime':
        text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
            img_name, img_text, model, label_to_explain)
    elif exp_method == 'torchray':
        text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
            img_name, img_text, model, label_to_explain)
    else:
        text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
            img_name, img_text, model, label_to_explain)

    img_exp_name, _ = os.path.splitext(img_exp)
    exp_text_visl = img_exp_name + '_text.png'


def profile_method(model_type, exp_method):
    stats_name = "stats/" + model_type + "_" + exp_method + "_" + device + ".stats"
    cProfile.run(f"predict('{model_type}', '{exp_method}')", stats_name)

def profile_all():
    model_list = ["MMBT", "LateFusion", "ViLBERT", "VisualBERT"]
    method_list = ["lime", "shap", "torchray"]
    try_profile(model_list, method_list)

def try_profile(model_list, method_list):
    error_msg = "error:\n"
    for model_type in model_list:
        for exp_method in method_list:
            try:
                profile_method(model_type, exp_method)
            except:
                error_msg = error_msg + model_type + "," + exp_method + "\n"
    print(error_msg)

if __name__ == "__main__":
    profile_all()
