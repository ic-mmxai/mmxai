from explanation_time_gpu import predict
from memory_profiler import profile

import torch
import sys

device = "gpu"

def mprofile(model_type, exp_method):
    mlog_name = "mlog/" + model_type + "_" + exp_method + "_" + device + ".mlog"
    with open(mlog_name, "w") as log_file:
        mprofile_predict = profile(func=predict, stream=log_file)
        mprofile_predict(model_type, exp_method)
        if device == "gpu":
            log_file.write("======== VRAM INFO ========\n")
            vram = torch.cuda.max_memory_reserved(device=torch.device("cuda:0"))
            log_file.write("cuda:0 reserved " + str(vram) + " bytes")
        else:
            log_file.write("======== CPU COMPUTATION, NO VRAM USAGE ========\n")

def profile_all():
    error_msg = "error:\n"
    for model_type in ["MMBT", "LateFusion", "ViLBERT", "VisualBERT"]:
        for exp_method in ["lime", "shap", "torchray"]:
            try:
                mprofile(model_type, exp_method)
            except:
                error_msg = error_msg + model_type + "," + exp_method + "\n"
    print(error_msg)

def profile_individual(model_type, exp_method):
    error_msg = "error:\n"
    try:
        mprofile(model_type, exp_method)
    except:
        error_msg = error_msg + model_type + "," + exp_method + "\n"
        print(error_msg)

if __name__ == "__main__":
    # profile_all()

    model_type = sys.argv[1]
    exp_method = sys.argv[2]

    profile_individual(model_type, exp_method)