from explanation_time_cpu import predict, device

import torch
import torch.autograd.profiler as profiler
from pytorch_memlab import LineProfiler
from memory_profiler import profile

# with profiler.profile(profile_memory=True, use_cuda=True) as prof:
#    predict("MMBT", "shap")

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# with LineProfiler(predict) as prof:
#     predict("MMBT", "lime")

# prof.display()

# @profile
# def profile(model_type, exp_method):
#     predict(model_type, exp_method)

print(predict.prepare_explanation)
# predict = profile(predict)

# predict("MMBT", "shap")

# print(torch.cuda.max_memory_reserved(device=torch.device("cuda:0")))