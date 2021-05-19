
    Sun May  9 11:30:10 2021    stats/LateFusion_lime_cpu.stats
    
             17561742 function calls (16093569 primitive calls) in 236.417 seconds
    
       Ordered by: internal time
       List reduced from 2201 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        36144  126.022    0.003  126.022    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
        77810   61.070    0.001   61.070    0.001 {built-in method conv2d}
         6024    7.429    0.001    7.429    0.001 {built-in method torch._C._nn.gelu}
        79818    5.323    0.000    5.323    0.000 {built-in method batch_norm}
        12048    5.175    0.000    5.175    0.000 {built-in method matmul}
          313    2.197    0.007    2.197    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
          502    2.159    0.004    2.159    0.004 {built-in method max_pool2d}
        39658    1.954    0.000  129.018    0.003 functional.py:1655(linear)
            1    1.845    1.845  226.085  226.085 lime_multimodal.py:359(data_labels)
    381520/502    1.716    0.000  221.963    0.442 module.py:710(_call_impl)
        25100    1.700    0.000   71.656    0.003 resnet.py:101(forward)
        75802    1.554    0.000    1.554    0.000 {built-in method relu_}
          797    1.419    0.002    1.419    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
         6024    0.935    0.000   40.695    0.007 hf_layers.py:122(forward)
        12550    0.788    0.000    0.788    0.000 {built-in method layer_norm}
            1    0.757    0.757    0.760    0.760 {skimage.segmentation._quickshift_cy._quickshift_cython}
            4    0.682    0.171    0.682    0.171 {method 'do_handshake' of '_ssl._SSLSocket' objects}
         6526    0.666    0.000    0.666    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
      1003678    0.662    0.000    0.663    0.000 module.py:758(__getattr__)
         3514    0.555    0.000    0.555    0.000 {built-in method addmm}
    
    
    Sun May  9 11:30:10 2021    stats/LateFusion_lime_gpu.stats
    
             17581761 function calls (16113588 primitive calls) in 30.373 seconds
    
       Ordered by: internal time
       List reduced from 2206 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        77810    2.994    0.000    2.994    0.000 {built-in method conv2d}
        79818    2.238    0.000    2.238    0.000 {built-in method batch_norm}
          313    2.157    0.007    2.157    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
            1    1.915    1.915   20.159   20.159 lime_multimodal.py:359(data_labels)
          797    1.469    0.002    1.469    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
    381520/502    1.403    0.000   15.007    0.030 module.py:710(_call_impl)
        36144    1.157    0.000    1.157    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
            1    0.847    0.847    0.850    0.850 {skimage.segmentation._quickshift_cy._quickshift_cython}
        75802    0.807    0.000    0.807    0.000 {built-in method relu_}
            4    0.675    0.169    0.675    0.169 {method 'do_handshake' of '_ssl._SSLSocket' objects}
        25100    0.670    0.000    8.926    0.000 resnet.py:101(forward)
        12048    0.655    0.000    0.655    0.000 {built-in method matmul}
        39658    0.608    0.000    2.246    0.000 functional.py:1655(linear)
      1003678    0.593    0.000    0.594    0.000 module.py:758(__getattr__)
           51    0.582    0.011   17.179    0.337 lime_mmf.py:16(multi_predict)
          502    0.500    0.001    0.500    0.001 {method 'resize' of 'ImagingCore' objects}
    484085/600    0.467    0.000    3.148    0.005 copy.py:128(deepcopy)
            4    0.395    0.099    0.395    0.099 {method 'read' of '_ssl._SSLSocket' objects}
        12550    0.337    0.000    0.337    0.000 {built-in method layer_norm}
            4    0.335    0.084    0.335    0.084 {method 'connect' of '_socket.socket' objects}
    
    
    Sun May  9 11:30:10 2021    stats/LateFusion_shap_cpu.stats
    
             13369061 function calls (12131368 primitive calls) in 100.739 seconds
    
       Ordered by: internal time
       List reduced from 3034 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        14040   51.525    0.004   51.525    0.004 {method 'matmul' of 'torch._C._TensorBase' objects}
        30225   24.547    0.001   24.547    0.001 {built-in method conv2d}
         2340    3.059    0.001    3.059    0.001 {built-in method torch._C._nn.gelu}
          313    2.273    0.007    2.273    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
         4680    1.995    0.000    1.995    0.000 {built-in method matmul}
        31005    1.963    0.000    1.963    0.000 {built-in method batch_norm}
          797    1.435    0.002    1.435    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
        15405    0.765    0.000   52.681    0.003 functional.py:1655(linear)
          195    0.703    0.004    0.703    0.004 {built-in method max_pool2d}
    148200/195    0.655    0.000   89.456    0.459 module.py:710(_call_impl)
         9750    0.632    0.000   28.500    0.003 resnet.py:101(forward)
        29445    0.596    0.000    0.596    0.000 {built-in method relu_}
            4    0.560    0.140    0.560    0.140 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    483763/65    0.452    0.000    3.017    0.046 copy.py:128(deepcopy)
         2340    0.354    0.000   16.389    0.007 hf_layers.py:122(forward)
            4    0.354    0.088    0.354    0.088 {method 'read' of '_ssl._SSLSocket' objects}
         4875    0.308    0.000    0.308    0.000 {built-in method layer_norm}
            4    0.276    0.069    0.276    0.069 {method 'connect' of '_socket.socket' objects}
         2535    0.263    0.000    0.263    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
         2210    0.254    0.000    0.254    0.000 {method '_set_from_file' of 'torch._C.FloatStorageBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/LateFusion_shap_gpu.stats
    
             13376329 function calls (12138636 primitive calls) in 18.896 seconds
    
       Ordered by: internal time
       List reduced from 3039 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          313    2.515    0.008    2.515    0.008 {method 'normal_' of 'torch._C._TensorBase' objects}
          797    1.861    0.002    1.861    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
        30225    1.204    0.000    1.204    0.000 {built-in method conv2d}
        31005    0.889    0.000    0.889    0.000 {built-in method batch_norm}
            4    0.678    0.169    0.678    0.169 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    148200/195    0.548    0.000    5.958    0.031 module.py:710(_call_impl)
    483763/65    0.479    0.000    3.216    0.049 copy.py:128(deepcopy)
        14040    0.458    0.000    0.458    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
            4    0.388    0.097    0.388    0.097 {method 'read' of '_ssl._SSLSocket' objects}
            4    0.351    0.088    0.351    0.088 {method 'connect' of '_socket.socket' objects}
        29445    0.318    0.000    0.318    0.000 {built-in method relu_}
         2210    0.286    0.000    0.286    0.000 {method '_set_from_file' of 'torch._C.FloatStorageBase' objects}
         9750    0.267    0.000    3.547    0.000 resnet.py:101(forward)
         4680    0.262    0.000    0.262    0.000 {built-in method matmul}
        15405    0.243    0.000    0.892    0.000 functional.py:1655(linear)
       393362    0.242    0.000    0.244    0.000 module.py:758(__getattr__)
          195    0.242    0.001    0.242    0.001 {method 'resize' of 'ImagingCore' objects}
            8    0.232    0.029    7.487    0.936 _explainer.py:279(_f_multimodal)
    55803/161    0.232    0.000    2.950    0.018 copy.py:226(_deepcopy_dict)
          232    0.227    0.001    0.227    0.001 {method 'sort' of 'numpy.ndarray' objects}
    
    
    Sun May  9 11:30:10 2021    stats/LateFusion_torchray_cpu.stats
    
             33900654 function calls (31581612 primitive calls) in 449.057 seconds
    
       Ordered by: internal time
       List reduced from 2057 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          800  141.400    0.177  141.400    0.177 {method 'run_backward' of 'torch._C._EngineBase' objects}
       243211  114.696    0.000  114.696    0.000 {built-in method conv2d}
       112968   86.980    0.001   86.980    0.001 {method 'matmul' of 'torch._C._TensorBase' objects}
       249471   16.816    0.000   16.816    0.000 {built-in method batch_norm}
        18828   11.644    0.001   11.644    0.001 {built-in method torch._C._nn.gelu}
        37656    5.972    0.000    5.972    0.000 {built-in method matmul}
        21197    5.611    0.000    5.611    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
         1569    5.498    0.004    5.498    0.004 {built-in method max_pool2d}
          800    5.310    0.007   16.849    0.021 extremal_perturbation.py:393(generate)
          800    5.100    0.006    5.100    0.006 {built-in method torch._C._nn.upsample_nearest2d}
        78450    4.556    0.000  144.782    0.002 resnet.py:101(forward)
    1192440/1569    4.335    0.000  274.313    0.175 module.py:710(_call_impl)
       123951    3.555    0.000   92.917    0.001 functional.py:1655(linear)
          800    3.323    0.004    3.323    0.004 {method 'sort' of 'torch._C._TensorBase' objects}
       236919    3.253    0.000    3.253    0.000 {built-in method relu_}
          313    2.039    0.007    2.039    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
      3124874    1.809    0.000    1.810    0.000 module.py:758(__getattr__)
        18828    1.591    0.000   33.794    0.002 hf_layers.py:122(forward)
        39225    1.537    0.000    1.537    0.000 {built-in method layer_norm}
          797    1.424    0.002    1.424    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/LateFusion_torchray_gpu.stats
    
             33921992 function calls (31602920 primitive calls) in 69.517 seconds
    
       Ordered by: internal time
       List reduced from 2064 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          800   10.821    0.014   10.821    0.014 {method 'run_backward' of 'torch._C._EngineBase' objects}
       243211    8.719    0.000    8.719    0.000 {built-in method conv2d}
       249471    6.224    0.000    6.224    0.000 {built-in method batch_norm}
    1192440/1569    4.294    0.000   42.426    0.027 module.py:710(_call_impl)
       112968    2.893    0.000    2.893    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
       236919    2.335    0.000    2.335    0.000 {built-in method relu_}
          313    2.168    0.007    2.168    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
         2370    2.093    0.001    2.093    0.001 {method 'cpu' of 'torch._C._TensorBase' objects}
        78450    2.007    0.000   25.928    0.000 resnet.py:101(forward)
      3124874    1.781    0.000    1.782    0.000 module.py:758(__getattr__)
       123951    1.659    0.000    5.796    0.000 functional.py:1655(linear)
        37656    1.589    0.000    1.589    0.000 {built-in method matmul}
          797    1.410    0.002    1.410    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
        18828    0.912    0.000    6.894    0.000 hf_layers.py:122(forward)
       249471    0.857    0.000    8.389    0.000 batchnorm.py:99(forward)
        39225    0.763    0.000    0.763    0.000 {built-in method layer_norm}
      1192441    0.633    0.000    0.633    0.000 {built-in method torch._C._get_tracing_state}
            4    0.625    0.156    0.625    0.156 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    490434/829    0.468    0.000    2.995    0.004 copy.py:128(deepcopy)
       249471    0.464    0.000    6.847    0.000 functional.py:1998(batch_norm)
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_lime_cpu.stats
    
             17800366 function calls (16284273 primitive calls) in 188.906 seconds
    
       Ordered by: internal time
       List reduced from 2402 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        36646   93.738    0.003   93.738    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
        77810   48.949    0.001   48.949    0.001 {built-in method conv2d}
        77810    5.860    0.000    5.860    0.000 {built-in method batch_norm}
         6526    5.071    0.001    5.071    0.001 {built-in method torch._C._nn.gelu}
        12048    4.946    0.000    4.946    0.000 {built-in method matmul}
          791    2.224    0.003    2.224    0.003 {method 'uniform_' of 'torch._C._TensorBase' objects}
          502    2.167    0.004    2.167    0.004 {built-in method max_pool2d}
          314    2.158    0.007    2.158    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
            1    1.893    1.893  178.173  178.173 lime_multimodal.py:359(data_labels)
        25100    1.641    0.000   59.949    0.002 resnet.py:101(forward)
    379010/502    1.598    0.000  173.927    0.346 module.py:710(_call_impl)
        38152    1.578    0.000   95.894    0.003 functional.py:1655(linear)
        75802    1.539    0.000    1.539    0.000 {built-in method relu_}
         6024    0.793    0.000   31.775    0.005 hf_layers.py:122(forward)
            1    0.757    0.757    0.760    0.760 {skimage.segmentation._quickshift_cy._quickshift_cython}
        13554    0.689    0.000    0.689    0.000 {built-in method layer_norm}
      1000591    0.633    0.000    0.634    0.000 module.py:758(__getattr__)
            4    0.626    0.156    0.626    0.156 {method 'do_handshake' of '_ssl._SSLSocket' objects}
         6526    0.569    0.000    0.569    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
          502    0.489    0.001    0.489    0.001 {method 'resize' of 'ImagingCore' objects}
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_lime_gpu.stats
    
             17814526 function calls (16298430 primitive calls) in 28.900 seconds
    
       Ordered by: internal time
       List reduced from 2422 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        77810    2.694    0.000    2.694    0.000 {built-in method conv2d}
          314    2.100    0.007    2.100    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
        77810    1.930    0.000    1.930    0.000 {built-in method batch_norm}
            1    1.775    1.775   17.782   17.782 lime_multimodal.py:359(data_labels)
         4657    1.536    0.000    1.537    0.000 {method 'to' of 'torch._C._TensorBase' objects}
          791    1.524    0.002    1.524    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
    379010/502    1.235    0.000   13.406    0.027 module.py:710(_call_impl)
        36646    0.999    0.000    0.999    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
        75802    0.740    0.000    0.740    0.000 {built-in method relu_}
            1    0.725    0.725    0.727    0.727 {skimage.segmentation._quickshift_cy._quickshift_cython}
        25100    0.617    0.000    8.039    0.000 resnet.py:101(forward)
            4    0.602    0.150    0.602    0.150 {method 'do_handshake' of '_ssl._SSLSocket' objects}
        12048    0.549    0.000    0.549    0.000 {built-in method matmul}
        38152    0.536    0.000    1.859    0.000 functional.py:1655(linear)
      1000591    0.505    0.000    0.506    0.000 module.py:758(__getattr__)
          502    0.461    0.001    0.461    0.001 {method 'resize' of 'ImagingCore' objects}
    506023/604    0.454    0.000    2.928    0.005 copy.py:128(deepcopy)
            4    0.367    0.092    0.367    0.092 {method 'read' of '_ssl._SSLSocket' objects}
           51    0.335    0.007   15.103    0.296 lime_mmf.py:16(multi_predict)
         2170    0.329    0.000    0.329    0.000 {method '_set_from_file' of 'torch._C.FloatStorageBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_shap_cpu.stats
    
             15059476 function calls (13682370 primitive calls) in 94.811 seconds
    
       Ordered by: internal time
       List reduced from 5784 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        14235   46.023    0.003   46.023    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
        30225   22.058    0.001   22.058    0.001 {built-in method conv2d}
         2535    2.465    0.001    2.465    0.001 {built-in method torch._C._nn.gelu}
        30225    2.315    0.000    2.315    0.000 {built-in method batch_norm}
         4680    2.289    0.000    2.289    0.000 {built-in method matmul}
          791    2.198    0.003    2.198    0.003 {method 'uniform_' of 'torch._C._TensorBase' objects}
          314    2.194    0.007    2.194    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
          195    0.763    0.004    0.763    0.004 {built-in method max_pool2d}
        14820    0.719    0.000   46.993    0.003 functional.py:1655(linear)
         9750    0.692    0.000   26.487    0.003 resnet.py:101(forward)
    147225/195    0.669    0.000   81.553    0.418 module.py:710(_call_impl)
        29445    0.609    0.000    0.609    0.000 {built-in method relu_}
            4    0.608    0.152    0.608    0.152 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    505701/69    0.477    0.000    3.104    0.045 copy.py:128(deepcopy)
            4    0.371    0.093    0.371    0.093 {method 'read' of '_ssl._SSLSocket' objects}
         2340    0.353    0.000   15.209    0.006 hf_layers.py:122(forward)
            4    0.305    0.076    0.305    0.076 {method 'connect' of '_socket.socket' objects}
         5265    0.296    0.000    0.296    0.000 {built-in method layer_norm}
         2535    0.269    0.000    0.269    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
       392117    0.264    0.000    0.265    0.000 module.py:758(__getattr__)
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_shap_gpu.stats
    
             15082809 function calls (13705498 primitive calls) in 17.689 seconds
    
       Ordered by: internal time
       List reduced from 5784 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          314    2.106    0.007    2.106    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
          791    1.516    0.002    1.516    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
        30225    1.074    0.000    1.074    0.000 {built-in method conv2d}
        30225    0.771    0.000    0.771    0.000 {built-in method batch_norm}
            4    0.650    0.163    0.650    0.163 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    147225/195    0.494    0.000    5.361    0.027 module.py:710(_call_impl)
    505701/69    0.455    0.000    2.954    0.043 copy.py:128(deepcopy)
        14235    0.401    0.000    0.401    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
            4    0.399    0.100    0.399    0.100 {method 'read' of '_ssl._SSLSocket' objects}
            4    0.317    0.079    0.317    0.079 {method 'connect' of '_socket.socket' objects}
        29445    0.295    0.000    0.295    0.000 {built-in method relu_}
         2170    0.275    0.000    0.275    0.000 {method '_set_from_file' of 'torch._C.FloatStorageBase' objects}
         9750    0.244    0.000    3.202    0.000 resnet.py:101(forward)
         4680    0.220    0.000    0.220    0.000 {built-in method matmul}
          257    0.215    0.001    0.215    0.001 {method 'sort' of 'numpy.ndarray' objects}
        14820    0.215    0.000    0.747    0.000 functional.py:1655(linear)
          195    0.213    0.001    0.213    0.001 {method 'resize' of 'ImagingCore' objects}
      1415246    0.208    0.000    0.391    0.000 {built-in method builtins.isinstance}
       392117    0.208    0.000    0.209    0.000 module.py:758(__getattr__)
         1901    0.204    0.000    0.225    0.000 ffi.py:149(__call__)
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_torchray_cpu.stats
    
             34158934 function calls (31767456 primitive calls) in 503.669 seconds
    
       Ordered by: internal time
       List reduced from 2586 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          800  198.894    0.249  198.894    0.249 {method 'run_backward' of 'torch._C._EngineBase' objects}
       243211  109.238    0.000  109.238    0.000 {built-in method conv2d}
       114537   90.031    0.001   90.031    0.001 {method 'matmul' of 'torch._C._TensorBase' objects}
       243195   16.675    0.000   16.675    0.000 {built-in method batch_norm}
        20397   11.521    0.001   11.521    0.001 {built-in method torch._C._nn.gelu}
        37656    6.348    0.000    6.348    0.000 {built-in method matmul}
        21197    5.456    0.000    5.456    0.000 {method 'softmax' of 'torch._C._TensorBase' objects}
         1569    5.393    0.003    5.393    0.003 {built-in method max_pool2d}
          800    5.316    0.007   16.157    0.020 extremal_perturbation.py:393(generate)
          800    4.795    0.006    4.795    0.006 {built-in method torch._C._nn.upsample_nearest2d}
        78450    4.503    0.000  138.847    0.002 resnet.py:101(forward)
    1184595/1569    4.301    0.000  270.805    0.173 module.py:710(_call_impl)
       119244    3.418    0.000   94.706    0.001 functional.py:1655(linear)
          800    3.279    0.004    3.279    0.004 {method 'sort' of 'torch._C._TensorBase' objects}
       236919    2.995    0.000    2.995    0.000 {built-in method relu_}
          314    2.021    0.006    2.021    0.006 {method 'normal_' of 'torch._C._TensorBase' objects}
      3115385    1.768    0.000    1.768    0.000 module.py:758(__getattr__)
        18828    1.738    0.000   34.756    0.002 hf_layers.py:122(forward)
        42363    1.598    0.000    1.598    0.000 {built-in method layer_norm}
          791    1.436    0.002    1.436    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/MMBT_torchray_gpu.stats
    
             36257443 function calls (33822822 primitive calls) in 78.224 seconds
    
       Ordered by: internal time
       List reduced from 2652 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          800   14.281    0.018   14.281    0.018 {method 'run_backward' of 'torch._C._EngineBase' objects}
       243211    7.542    0.000    7.542    0.000 {built-in method conv2d}
         8000    5.831    0.001    5.831    0.001 {built-in method masked_select}
       243195    5.228    0.000    5.228    0.000 {built-in method batch_norm}
    1184595/1569    3.439    0.000   37.017    0.024 module.py:710(_call_impl)
       114537    2.810    0.000    2.810    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
          314    2.088    0.007    2.088    0.007 {method 'normal_' of 'torch._C._TensorBase' objects}
       236919    2.026    0.000    2.026    0.000 {built-in method relu_}
        78450    1.728    0.000   22.306    0.000 resnet.py:101(forward)
          791    1.620    0.002    1.620    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
        12132    1.604    0.000    1.605    0.000 {method 'to' of 'torch._C._TensorBase' objects}
        37656    1.552    0.000    1.552    0.000 {built-in method matmul}
       119244    1.437    0.000    5.018    0.000 functional.py:1655(linear)
      3115385    1.363    0.000    1.364    0.000 module.py:758(__getattr__)
            4    1.010    0.253    1.010    0.253 {method 'do_handshake' of '_ssl._SSLSocket' objects}
         8000    0.864    0.000    9.225    0.001 _tensor_str.py:74(__init__)
        25415    0.862    0.000    0.862    0.000 {built-in method cat}
        42363    0.805    0.000    0.805    0.000 {built-in method layer_norm}
        18828    0.786    0.000    6.258    0.000 hf_layers.py:122(forward)
       243195    0.770    0.000    7.056    0.000 batchnorm.py:99(forward)
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_lime_cpu.stats
    
             23613740 function calls (21279609 primitive calls) in 4230.587 seconds
    
       Ordered by: internal time
       List reduced from 2499 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        53714 3388.417    0.063 3388.417    0.063 {built-in method conv2d}
        91364  240.532    0.003  240.532    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
          502  177.159    0.353  177.159    0.353 {built-in method torch._ops.torchvision.roi_pool}
        52208  160.725    0.003  160.725    0.003 {built-in method batch_norm}
        16566   48.155    0.003 3591.447    0.217 modeling_frcnn.py:931(forward)
          502   35.725    0.071   35.725    0.071 {built-in method max_pool2d}
        50200   35.553    0.001   35.553    0.001 {built-in method relu_}
         4016   26.348    0.007   26.348    0.007 {built-in method addmm}
        52208   20.170    0.000 3523.396    0.067 modeling_frcnn.py:760(forward)
        15562   12.418    0.001   12.418    0.001 {built-in method torch._C._nn.gelu}
    386540/1004    9.163    0.000 4200.394    4.184 module.py:710(_call_impl)
        30120    9.081    0.000    9.081    0.000 {built-in method matmul}
          502    7.411    0.015    7.411    0.015 {built-in method flip}
          502    7.305    0.015    7.305    0.015 {built-in method torch._C._nn.upsample_bilinear2d}
         2510    6.997    0.003 3602.487    1.435 container.py:115(forward)
        95380    3.325    0.000  271.147    0.003 functional.py:1655(linear)
          502    2.638    0.005    2.638    0.005 {method 'mean' of 'torch._C._TensorBase' objects}
          210    2.311    0.011    2.311    0.011 {method 'normal_' of 'torch._C._TensorBase' objects}
            1    1.814    1.814 4214.196 4214.196 lime_multimodal.py:359(data_labels)
          502    1.574    0.003  178.743    0.356 roi_pool.py:49(forward)
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_lime_gpu.stats
    
             23610361 function calls (21276230 primitive calls) in 3970.696 seconds
    
       Ordered by: internal time
       List reduced from 2498 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        53714 3372.312    0.063 3372.312    0.063 {built-in method conv2d}
          502  196.983    0.392  196.983    0.392 {built-in method torch._ops.torchvision.roi_pool}
        52208  165.494    0.003  165.494    0.003 {built-in method batch_norm}
        16566   47.175    0.003 3567.706    0.215 modeling_frcnn.py:931(forward)
        50200   36.179    0.001   36.179    0.001 {built-in method relu_}
          502   35.861    0.071   35.861    0.071 {built-in method max_pool2d}
         4016   26.044    0.006   26.044    0.006 {built-in method addmm}
        52208   15.147    0.000 3507.530    0.067 modeling_frcnn.py:760(forward)
    386540/1004    8.466    0.000 3940.768    3.925 module.py:710(_call_impl)
          502    7.651    0.015    7.651    0.015 {built-in method torch._C._nn.upsample_bilinear2d}
         2510    7.481    0.003 3578.952    1.426 container.py:115(forward)
          502    7.275    0.014    7.275    0.014 {built-in method flip}
          502    2.633    0.005    2.633    0.005 {method 'mean' of 'torch._C._TensorBase' objects}
        91364    2.520    0.000    2.520    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
          210    1.912    0.009    1.912    0.009 {method 'normal_' of 'torch._C._TensorBase' objects}
            1    1.820    1.820 3954.895 3954.895 lime_multimodal.py:359(data_labels)
         1004    1.503    0.001    1.503    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
          502    1.476    0.003 3203.822    6.382 modeling_frcnn.py:1420(forward)
         6248    1.465    0.000    1.465    0.000 {method 'clamp_' of 'torch._C._TensorBase' objects}
        30120    1.396    0.000    1.396    0.000 {built-in method matmul}
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_shap_cpu.stats
    
             15696013 function calls (14062769 primitive calls) in 1496.103 seconds
    
       Ordered by: internal time
       List reduced from 3159 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        20865 1178.472    0.056 1178.472    0.056 {built-in method conv2d}
        35490   93.911    0.003   93.911    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
          195   62.968    0.323   62.968    0.323 {built-in method torch._ops.torchvision.roi_pool}
        20280   60.367    0.003   60.367    0.003 {built-in method batch_norm}
         6435   17.568    0.003 1250.221    0.194 modeling_frcnn.py:931(forward)
        19500   12.737    0.001   12.737    0.001 {built-in method relu_}
          195   11.093    0.057   11.093    0.057 {built-in method max_pool2d}
         1560    9.040    0.006    9.040    0.006 {built-in method addmm}
        20280    6.752    0.000 1227.689    0.061 modeling_frcnn.py:760(forward)
         6045    4.840    0.001    4.840    0.001 {built-in method torch._C._nn.gelu}
        11700    3.541    0.000    3.541    0.000 {built-in method matmul}
    150150/390    3.295    0.000 1478.751    3.792 module.py:710(_call_impl)
          195    2.908    0.015    2.908    0.015 {built-in method flip}
          195    2.806    0.014    2.806    0.014 {built-in method torch._C._nn.upsample_bilinear2d}
          975    2.622    0.003 1254.283    1.286 container.py:115(forward)
          210    2.330    0.011    2.330    0.011 {method 'normal_' of 'torch._C._TensorBase' objects}
        37050    1.302    0.000  104.625    0.003 functional.py:1655(linear)
          195    0.915    0.005    0.915    0.005 {method 'mean' of 'torch._C._TensorBase' objects}
            3    0.716    0.239    0.716    0.239 {method 'do_handshake' of '_ssl._SSLSocket' objects}
        12285    0.594    0.000    0.594    0.000 {built-in method layer_norm}
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_shap_gpu.stats
    
             15705018 function calls (14071665 primitive calls) in 1370.230 seconds
    
       Ordered by: internal time
       List reduced from 3170 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        20865 1157.346    0.055 1157.346    0.055 {built-in method conv2d}
          195   61.611    0.316   61.611    0.316 {built-in method torch._ops.torchvision.roi_pool}
        20280   60.117    0.003   60.117    0.003 {built-in method batch_norm}
         6435   17.290    0.003 1228.197    0.191 modeling_frcnn.py:931(forward)
        19500   12.654    0.001   12.654    0.001 {built-in method relu_}
          195   10.926    0.056   10.926    0.056 {built-in method max_pool2d}
         1560    8.920    0.006    8.920    0.006 {built-in method addmm}
        20280    6.968    0.000 1206.517    0.059 modeling_frcnn.py:760(forward)
    150150/390    3.472    0.000 1353.446    3.470 module.py:710(_call_impl)
          975    3.049    0.003 1232.832    1.264 container.py:115(forward)
          195    2.919    0.015    2.919    0.015 {built-in method torch._C._nn.upsample_bilinear2d}
          195    2.853    0.015    2.853    0.015 {built-in method flip}
          210    1.884    0.009    1.884    0.009 {method 'normal_' of 'torch._C._TensorBase' objects}
        35490    0.972    0.000    0.972    0.000 {method 'matmul' of 'torch._C._TensorBase' objects}
          195    0.872    0.004    0.872    0.004 {method 'mean' of 'torch._C._TensorBase' objects}
            3    0.824    0.275    0.824    0.275 {method 'do_handshake' of '_ssl._SSLSocket' objects}
         2920    0.625    0.000    0.625    0.000 {method 'clamp_' of 'torch._C._TensorBase' objects}
          390    0.579    0.001    0.579    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
          195    0.579    0.003   62.194    0.319 roi_pool.py:49(forward)
        11700    0.541    0.000    0.541    0.000 {built-in method matmul}
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_torchray_cpu.stats
    
             2244263483 function calls (2019345921 primitive calls) in 11752.360 seconds
    
       Ordered by: internal time
       List reduced from 2354 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       167899 8161.694    0.049 8161.694    0.049 {built-in method conv2d}
       163176  789.209    0.005  789.209    0.005 {built-in method batch_norm}
         1569  384.905    0.245  384.905    0.245 {built-in method torch._ops.torchvision.roi_pool}
       185206  352.595    0.002  352.595    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
         9618  246.822    0.026  246.822    0.026 {method 'normal_' of 'torch._C._TensorBase' objects}
        51777  198.376    0.004 9389.182    0.181 modeling_frcnn.py:931(forward)
       285558  168.295    0.001  168.295    0.001 {method 'matmul' of 'torch._C._TensorBase' objects}
       156900  167.631    0.001  167.631    0.001 {built-in method relu_}
    102996430/88688   92.774    0.000  458.161    0.005 copy.py:128(deepcopy)
       163176   78.032    0.000 9025.930    0.055 modeling_frcnn.py:760(forward)
      1004160   59.710    0.000   61.100    0.000 serialization.py:818(load_tensor)
    1208130/3138   47.394    0.000 10206.229    3.252 module.py:710(_call_impl)
         7845   42.502    0.005 9451.769    1.205 container.py:115(forward)
      1004882   42.005    0.000   42.005    0.000 {method 'copy_' of 'torch._C._TensorBase' objects}
        12552   35.575    0.003   35.575    0.003 {built-in method addmm}
       416845   34.325    0.000  105.880    0.000 module.py:917(_load_from_state_dict)
    200978739   29.695    0.000   54.033    0.000 {built-in method builtins.isinstance}
    78496610/41579210   26.901    0.000   43.897    0.000 typing.py:720(__hash__)
    8451478/4223243   26.091    0.000  142.854    0.000 copy.py:258(_reconstruct)
    267162268   25.916    0.000   25.916    0.000 {method 'startswith' of 'str' objects}
    
    
    Sun May  9 11:30:10 2021    stats/ViLBERT_torchray_gpu.stats
    
             2244320570 function calls (2019401795 primitive calls) in 1983.931 seconds
    
       Ordered by: internal time
       List reduced from 2850 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         3140  360.328    0.115  360.665    0.115 modeling_frcnn.py:147(_clip_box)
       185206  350.021    0.002  350.021    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
         9618  212.077    0.022  212.077    0.022 {method 'normal_' of 'torch._C._TensorBase' objects}
      1004160  101.120    0.000  102.487    0.000 serialization.py:818(load_tensor)
    102996430/88688   91.204    0.000  449.391    0.005 copy.py:128(deepcopy)
      2049011   54.255    0.000   54.256    0.000 {method 'to' of 'torch._C._TensorBase' objects}
       416845   33.786    0.000   95.867    0.000 module.py:917(_load_from_state_dict)
      1004882   32.938    0.000   32.938    0.000 {method 'copy_' of 'torch._C._TensorBase' objects}
    200980595   29.493    0.000   53.534    0.000 {built-in method builtins.isinstance}
    78496621/41579216   26.373    0.000   42.717    0.000 typing.py:720(__hash__)
    267162338   25.604    0.000   25.604    0.000 {method 'startswith' of 'str' objects}
    8451479/4223244   25.514    0.000  140.417    0.000 copy.py:258(_reconstruct)
    12072557/94288   24.986    0.000  432.233    0.005 copy.py:226(_deepcopy_dict)
    237301831   20.001    0.000   20.001    0.000 {method 'get' of 'dict' objects}
      4614675   19.356    0.000  126.607    0.000 _utils.py:332(get_value_kind)
    97002576/55469677   15.281    0.000   27.203    0.000 {built-in method builtins.hash}
    23120512/23120509   14.007    0.000   76.114    0.000 typing.py:255(inner)
    3619451/1928446   13.883    0.000   34.030    0.000 base.py:405(_re_parent)
    13935053/4705703   13.473    0.000   24.222    0.000 typing.py:711(__eq__)
     69749667   12.880    0.000   12.881    0.000 {built-in method _abc._abc_instancecheck}
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_lime_cpu.stats
    
             19691144 function calls (17611438 primitive calls) in 3935.058 seconds
    
       Ordered by: internal time
       List reduced from 2303 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        53714 3230.563    0.060 3230.563    0.060 {built-in method conv2d}
          502  170.735    0.340  170.735    0.340 {built-in method torch._ops.torchvision.roi_pool}
        36646  158.608    0.004  158.608    0.004 {method 'matmul' of 'torch._C._TensorBase' objects}
        52208  145.883    0.003  145.883    0.003 {built-in method batch_norm}
        16566   43.972    0.003 3408.497    0.206 modeling_frcnn.py:931(forward)
          502   35.422    0.071   35.422    0.071 {built-in method max_pool2d}
        50200   33.947    0.001   33.947    0.001 {built-in method relu_}
         3514   24.333    0.007   24.333    0.007 {built-in method addmm}
        52208   11.234    0.000 3343.550    0.064 modeling_frcnn.py:760(forward)
         6526    9.497    0.001    9.497    0.001 {built-in method torch._C._nn.gelu}
    245478/1004    7.825    0.000 3906.845    3.891 module.py:710(_call_impl)
        12048    7.409    0.001    7.409    0.001 {built-in method matmul}
          502    7.209    0.014    7.209    0.014 {built-in method flip}
          502    7.195    0.014    7.195    0.014 {built-in method torch._C._nn.upsample_bilinear2d}
         2510    6.295    0.003 3418.046    1.362 container.py:115(forward)
          502    2.547    0.005    2.547    0.005 {method 'mean' of 'torch._C._TensorBase' objects}
            1    1.801    1.801 3920.230 3920.230 lime_multimodal.py:359(data_labels)
        40160    1.791    0.000  185.182    0.005 functional.py:1655(linear)
         1004    1.481    0.001    1.481    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
         6240    1.462    0.000    1.462    0.000 {method 'clamp_' of 'torch._C._TensorBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_lime_gpu.stats
    
             19688506 function calls (17608800 primitive calls) in 3911.912 seconds
    
       Ordered by: internal time
       List reduced from 2308 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        53714 3370.032    0.063 3370.032    0.063 {built-in method conv2d}
          502  173.259    0.345  173.259    0.345 {built-in method torch._ops.torchvision.roi_pool}
        52208  146.574    0.003  146.574    0.003 {built-in method batch_norm}
        16566   45.588    0.003 3552.595    0.214 modeling_frcnn.py:931(forward)
          502   35.737    0.071   35.737    0.071 {built-in method max_pool2d}
        50200   35.545    0.001   35.545    0.001 {built-in method relu_}
         3514   26.651    0.008   26.651    0.008 {built-in method addmm}
        52208   14.352    0.000 3485.148    0.067 modeling_frcnn.py:760(forward)
          502    7.650    0.015    7.650    0.015 {built-in method torch._C._nn.upsample_bilinear2d}
          502    7.417    0.015    7.417    0.015 {built-in method flip}
    245478/1004    7.289    0.000 3882.492    3.867 module.py:710(_call_impl)
         2510    5.990    0.002 3561.490    1.419 container.py:115(forward)
          502    2.592    0.005    2.592    0.005 {method 'mean' of 'torch._C._TensorBase' objects}
            1    1.839    1.839 3897.370 3897.370 lime_multimodal.py:359(data_labels)
          502    1.658    0.003 3196.773    6.368 modeling_frcnn.py:1420(forward)
          502    1.526    0.003  174.796    0.348 roi_pool.py:49(forward)
         1004    1.486    0.001    1.486    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
         6284    1.484    0.000    1.484    0.000 {method 'clamp_' of 'torch._C._TensorBase' objects}
          606    1.043    0.002    1.043    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
           92    1.040    0.011    1.040    0.011 {method 'normal_' of 'torch._C._TensorBase' objects}
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_shap_cpu.stats
    
             13591212 function calls (12108600 primitive calls) in 1072.178 seconds
    
       Ordered by: internal time
       List reduced from 3142 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        20865  846.500    0.041  846.500    0.041 {built-in method conv2d}
          195   53.473    0.274   53.473    0.274 {built-in method torch._ops.torchvision.roi_pool}
        20280   48.773    0.002   48.773    0.002 {built-in method batch_norm}
        14235   46.559    0.003   46.559    0.003 {method 'matmul' of 'torch._C._TensorBase' objects}
         6435   14.603    0.002  907.552    0.141 modeling_frcnn.py:931(forward)
          195   10.205    0.052   10.205    0.052 {built-in method max_pool2d}
        19500   10.097    0.001   10.097    0.001 {built-in method relu_}
         1365    6.105    0.004    6.105    0.004 {built-in method addmm}
        20280    4.493    0.000  886.955    0.044 modeling_frcnn.py:760(forward)
    95355/390    2.958    0.000 1058.455    2.714 module.py:710(_call_impl)
         2535    2.784    0.001    2.784    0.001 {built-in method torch._C._nn.gelu}
         4680    2.660    0.001    2.660    0.001 {built-in method matmul}
          195    2.581    0.013    2.581    0.013 {built-in method torch._C._nn.upsample_bilinear2d}
          975    2.389    0.002  911.212    0.935 container.py:115(forward)
          195    2.250    0.012    2.250    0.012 {built-in method flip}
           92    1.130    0.012    1.130    0.012 {method 'normal_' of 'torch._C._TensorBase' objects}
          606    1.016    0.002    1.016    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
          195    0.808    0.004    0.808    0.004 {method 'mean' of 'torch._C._TensorBase' objects}
          390    0.557    0.001    0.557    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
        15600    0.495    0.000   53.291    0.003 functional.py:1655(linear)
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_shap_gpu.stats
    
             13426664 function calls (11959975 primitive calls) in 1262.016 seconds
    
       Ordered by: internal time
       List reduced from 3147 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        20009 1078.956    0.054 1078.956    0.054 {built-in method conv2d}
          187   56.874    0.304   56.874    0.304 {built-in method torch._ops.torchvision.roi_pool}
        19448   48.649    0.003   48.649    0.003 {built-in method batch_norm}
         6171   15.614    0.003 1137.749    0.184 modeling_frcnn.py:931(forward)
        18700   11.872    0.001   11.872    0.001 {built-in method relu_}
          187    9.972    0.053    9.972    0.053 {built-in method max_pool2d}
         1309    8.213    0.006    8.213    0.006 {built-in method addmm}
        19448    4.755    0.000 1115.336    0.057 modeling_frcnn.py:760(forward)
          187    2.839    0.015    2.839    0.015 {built-in method torch._C._nn.upsample_bilinear2d}
          187    2.747    0.015    2.747    0.015 {built-in method flip}
    91443/374    2.464    0.000 1246.588    3.333 module.py:710(_call_impl)
          935    1.949    0.002 1140.679    1.220 container.py:115(forward)
          606    1.057    0.002    1.057    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
           92    1.018    0.011    1.018    0.011 {method 'normal_' of 'torch._C._TensorBase' objects}
          187    0.826    0.004    0.826    0.004 {method 'mean' of 'torch._C._TensorBase' objects}
            3    0.647    0.216    0.647    0.216 {method 'read' of '_ssl._SSLSocket' objects}
         2748    0.567    0.000    0.567    0.000 {method 'clamp_' of 'torch._C._TensorBase' objects}
          374    0.549    0.001    0.549    0.001 {method 'sort' of 'torch._C._TensorBase' objects}
    484905/93    0.441    0.000    2.817    0.030 copy.py:128(deepcopy)
            3    0.424    0.141    0.424    0.141 {method 'do_handshake' of '_ssl._SSLSocket' objects}
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_torchray_cpu.stats
    
             2233992684 function calls (2009691824 primitive calls) in 12126.418 seconds
    
       Ordered by: internal time
       List reduced from 2148 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       167899 8409.943    0.050 8409.943    0.050 {built-in method conv2d}
       163176  888.444    0.005  888.444    0.005 {built-in method batch_norm}
         1569  387.647    0.247  387.647    0.247 {built-in method torch._ops.torchvision.roi_pool}
       185630  354.981    0.002  354.981    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
         9500  245.834    0.026  245.834    0.026 {method 'normal_' of 'torch._C._TensorBase' objects}
        51777  219.946    0.004 9824.817    0.190 modeling_frcnn.py:931(forward)
       156900  169.649    0.001  169.649    0.001 {built-in method relu_}
       163176  142.562    0.001 9437.727    0.058 modeling_frcnn.py:760(forward)
       114537  103.220    0.001  103.220    0.001 {method 'matmul' of 'torch._C._TensorBase' objects}
    102938424/88665   92.871    0.000  460.823    0.005 copy.py:128(deepcopy)
      1004160   60.194    0.000   61.582    0.000 serialization.py:818(load_tensor)
    767241/3138   51.014    0.000 10572.670    3.369 module.py:710(_call_impl)
         7845   48.675    0.006 9896.198    1.261 container.py:115(forward)
      1004568   40.730    0.000   40.730    0.000 {method 'copy_' of 'torch._C._TensorBase' objects}
        10983   36.476    0.003   36.476    0.003 {built-in method addmm}
       416231   34.519    0.000  104.957    0.000 module.py:917(_load_from_state_dict)
    200903826   29.919    0.000   54.512    0.000 {built-in method builtins.isinstance}
    78435287/41546743   27.081    0.000   44.201    0.000 typing.py:720(__hash__)
    8443050/4221525   26.539    0.000  143.091    0.000 copy.py:258(_reconstruct)
    266865945   26.032    0.000   26.032    0.000 {method 'startswith' of 'str' objects}
    
    
    Sun May  9 11:30:10 2021    stats/VisualBERT_torchray_gpu.stats
    
             2234002570 function calls (2009701710 primitive calls) in 1937.285 seconds
    
       Ordered by: internal time
       List reduced from 2153 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         3140  365.241    0.116  365.600    0.116 modeling_frcnn.py:147(_clip_box)
       185630  353.272    0.002  353.272    0.002 {method 'uniform_' of 'torch._C._TensorBase' objects}
         9500  212.726    0.022  212.726    0.022 {method 'normal_' of 'torch._C._TensorBase' objects}
    102938424/88665   92.602    0.000  456.303    0.005 copy.py:128(deepcopy)
      1004160   59.394    0.000   60.761    0.000 serialization.py:818(load_tensor)
      2043988   54.419    0.000   54.419    0.000 {method 'to' of 'torch._C._TensorBase' objects}
      1004568   41.857    0.000   41.857    0.000 {method 'copy_' of 'torch._C._TensorBase' objects}
       416231   34.438    0.000  105.880    0.000 module.py:917(_load_from_state_dict)
    200903814   29.666    0.000   54.111    0.000 {built-in method builtins.isinstance}
    78435287/41546743   26.683    0.000   43.267    0.000 typing.py:720(__hash__)
    8443050/4221525   25.998    0.000  142.208    0.000 copy.py:258(_reconstruct)
    266865655   25.976    0.000   25.976    0.000 {method 'startswith' of 'str' objects}
    12066734/94260   25.392    0.000  439.025    0.005 copy.py:226(_deepcopy_dict)
    237196021   20.148    0.000   20.148    0.000 {method 'get' of 'dict' objects}
      4611068   19.881    0.000  128.682    0.000 _utils.py:332(get_value_kind)
    96926768/55426343   15.484    0.000   27.575    0.000 {built-in method builtins.hash}
     23102472   14.282    0.000   77.126    0.000 typing.py:255(inner)
    3618839/1928232   14.171    0.000   34.536    0.000 base.py:405(_re_parent)
    13924212/4702076   13.587    0.000   24.426    0.000 typing.py:711(__eq__)
     69712901   13.058    0.000   13.058    0.000 {built-in method _abc._abc_instancecheck}
    
    
    Sun May  9 11:30:10 2021    stats/inpainting.stats
    
             18637 function calls (18326 primitive calls) in 1.189 seconds
    
       Ordered by: internal time
       List reduced from 878 to 20 due to restriction <20>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.645    0.645    0.645    0.645 {inpaint}
            1    0.173    0.173    0.173    0.173 {readNet}
            1    0.133    0.133    0.133    0.133 {method 'forward' of 'cv2.dnn_Net' objects}
           42    0.084    0.002    0.084    0.002 {method 'read' of '_ssl._SSLSocket' objects}
            1    0.042    0.042    0.042    0.042 {built-in method _socket.getaddrinfo}
            1    0.019    0.019    0.019    0.019 {method 'do_handshake' of '_ssl._SSLSocket' objects}
            1    0.014    0.014    0.014    0.014 {blobFromImage}
            1    0.013    0.013    0.013    0.013 smart_text_removal.py:220(decodeBoundingBoxes)
            1    0.013    0.013    0.013    0.013 {method 'connect' of '_socket.socket' objects}
            1    0.008    0.008    1.188    1.188 <string>:1(<module>)
            1    0.006    0.006    0.006    0.006 {method 'load_verify_locations' of '_ssl._SSLContext' objects}
            2    0.006    0.003    0.006    0.003 {method 'decode' of 'ImagingDecoder' objects}
           22    0.005    0.000    0.005    0.000 smart_text_removal.py:195(isOnRight)
           22    0.005    0.000    0.005    0.000 smart_text_removal.py:205(isAbove)
           21    0.002    0.000    0.002    0.000 {built-in method marshal.loads}
           79    0.001    0.000    0.001    0.000 {built-in method __new__ of type object at 0x561d4abceac0}
            1    0.001    0.001    0.013    0.013 smart_text_removal.py:130(generateTextMask)
           12    0.001    0.000    0.001    0.000 {method 'reduce' of 'numpy.ufunc' objects}
            3    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}
            3    0.001    0.000    0.001    0.000 {built-in method _imp.create_dynamic}
