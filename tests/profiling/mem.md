    ========== LateFusion_lime_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.8 MiB    460.8 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1854.4 MiB   1393.6 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.8 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1854.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1854.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1854.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1854.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1854.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1854.4 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   1868.2 MiB     13.9 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   1854.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1868.2 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1868.2 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== LateFusion_lime_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.3 MiB    454.3 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3087.4 MiB   2633.1 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.3 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3087.4 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3087.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3087.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3087.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3087.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3087.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3087.4 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   3091.4 MiB      3.9 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   3087.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3091.4 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3091.4 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1021313024 bytes
    
     
    
    ========== LateFusion_shap_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    459.9 MiB    459.9 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1853.4 MiB   1393.5 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    459.9 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1853.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1853.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1853.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1853.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1853.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   1868.4 MiB     15.0 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   1853.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1868.4 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1868.4 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== LateFusion_shap_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.7 MiB    454.7 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3089.7 MiB   2634.9 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.7 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3089.7 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3089.7 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3089.7 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3089.7 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3089.7 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3089.7 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   3105.5 MiB     15.8 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   3089.7 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3105.5 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3105.5 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1021313024 bytes
    
     
    
    ========== LateFusion_torchray_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.3 MiB    460.3 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   2075.1 MiB   1614.8 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.3 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   2075.1 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   2075.1 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   2075.1 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   2075.1 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   2075.1 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   2075.1 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   2075.1 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   2083.7 MiB      8.6 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   2075.1 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   2083.7 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   2083.7 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== LateFusion_torchray_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    453.9 MiB    453.9 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3079.2 MiB   2625.4 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    453.9 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3079.2 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3079.2 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3079.2 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3079.2 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3079.2 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3079.2 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3079.2 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   3079.2 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   3111.7 MiB     32.5 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   3079.2 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3111.7 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3111.7 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1228931072 bytes
    
     
    
    ========== MMBT_lime_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.5 MiB    460.5 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   2072.5 MiB   1612.0 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.5 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   2072.5 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   2072.5 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   2072.5 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   2072.5 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   2072.5 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   2072.5 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   2085.5 MiB     13.0 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   2072.5 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   2085.5 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   2085.5 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== MMBT_lime_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    455.1 MiB    455.1 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3088.1 MiB   2633.0 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    455.1 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3088.1 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3088.1 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3088.1 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3088.1 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3088.1 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3088.1 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3088.1 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   3093.7 MiB      5.6 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   3088.1 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3093.7 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3093.7 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 981467136 bytes
    
     
    
    ========== MMBT_shap_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    459.3 MiB    459.3 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   2071.0 MiB   1611.7 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    459.3 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   2071.0 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   2071.0 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   2071.0 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   2071.0 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   2071.0 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   2090.4 MiB     19.4 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   2071.0 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   2090.4 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   2090.4 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== MMBT_shap_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.3 MiB    454.3 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3079.7 MiB   2625.4 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.3 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3079.7 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3079.7 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3079.7 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3079.7 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3079.7 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3079.7 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   3096.5 MiB     16.7 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   3079.7 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3096.5 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3096.5 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 981467136 bytes
    
     
    
    ========== MMBT_torchray_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.9 MiB    460.9 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1847.4 MiB   1386.5 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.9 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1847.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1847.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1847.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1847.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1847.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1847.4 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   1847.4 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   1908.3 MiB     61.0 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   1847.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1908.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1908.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== MMBT_torchray_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.0 MiB    454.0 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3082.2 MiB   2628.1 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.0 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3082.2 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3082.2 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3082.2 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3082.2 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3082.2 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3082.2 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3082.2 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   3082.2 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   3089.4 MiB      7.2 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   3082.2 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3089.4 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3089.4 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1375731712 bytes
    
     
    
    ========== ViLBERT_lime_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.8 MiB    460.8 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1918.8 MiB   1458.0 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.8 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1918.8 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1918.8 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1918.8 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1918.8 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1918.8 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1918.8 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   1942.5 MiB     23.7 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   1918.8 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1942.5 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1942.5 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== ViLBERT_lime_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.7 MiB    454.7 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3458.6 MiB   3003.9 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.7 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3458.6 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3458.6 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3458.6 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3458.6 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3458.6 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3458.6 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3458.6 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   3486.6 MiB     28.1 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   3458.6 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3486.6 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3486.6 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1191182336 bytes
    
     
    
    ========== ViLBERT_shap_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.0 MiB    460.0 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1917.7 MiB   1457.7 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.0 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1917.7 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1917.7 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1917.7 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1917.7 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1917.7 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   2261.3 MiB    343.6 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   1917.7 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   2261.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   2261.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== ViLBERT_shap_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.0 MiB    454.0 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3429.4 MiB   2975.4 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.0 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3429.4 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3429.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3429.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3429.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3429.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3429.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   4028.7 MiB    599.3 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   3429.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   4028.7 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   4028.7 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 1191182336 bytes
    
     
    
    ========== ViLBERT_torchray_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.5 MiB    460.5 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1918.5 MiB   1457.9 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.5 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1918.5 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1918.5 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1918.5 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1918.5 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1918.5 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1918.5 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   1918.5 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   2162.3 MiB    243.8 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   1918.5 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   2162.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   2162.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== ViLBERT_torchray_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    463.6 MiB    463.6 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   2920.7 MiB   2457.1 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    463.6 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   2920.7 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   2920.7 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   2920.7 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   2920.7 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   2920.7 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   2920.7 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   2920.7 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   2920.7 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   3244.8 MiB    324.0 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   2920.7 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3244.8 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3244.8 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 3602907136 bytes
    
     
    
    ========== VisualBERT_lime_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    459.9 MiB    459.9 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1511.1 MiB   1051.2 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    459.9 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1511.1 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1511.1 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1511.1 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1511.1 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1511.1 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1511.1 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   1560.3 MiB     49.2 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   1511.1 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1560.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1560.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== VisualBERT_lime_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    453.8 MiB    453.8 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3833.1 MiB   3379.2 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    453.8 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3833.1 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3833.1 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3833.1 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3833.1 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3833.1 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3833.1 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   3833.1 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33   3916.1 MiB     83.1 MiB           2           text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34   3833.1 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3916.1 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3916.1 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 610271232 bytes
    
     
    
    ========== VisualBERT_shap_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.2 MiB    460.2 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1545.4 MiB   1085.2 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.2 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1545.4 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1545.4 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1545.4 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1545.4 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1545.4 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   1697.3 MiB    151.9 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   1545.4 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1697.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1697.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== VisualBERT_shap_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    454.1 MiB    454.1 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   3858.5 MiB   3404.4 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    454.1 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   3858.5 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   3858.5 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   3858.5 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   3858.5 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   3858.5 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   3858.5 MiB      0.0 MiB           1       if exp_method == 'shap':
        30   4482.9 MiB    624.4 MiB           2           text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31   3858.5 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32                                             elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35                                             elif exp_method == 'torchray':
        36                                                 text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37                                                     img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   4482.9 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   4482.9 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 610271232 bytes
    
     
    
    ========== VisualBERT_torchray_cpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
        10    460.2 MiB    460.2 MiB           1   def predict(model_type,
        11                                                     exp_method,
        12                                                     user_model="no_model",
        13                                                     img_name="profiling.png",
        14                                                     img_text="kill the jews and muslims for being anti-white",
        15                                                     model_path=None,
        16                                                     exp_direction="encourage"):
        17                                         
        18   1521.0 MiB   1060.9 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        19    460.2 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        20                                         
        21   1521.0 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   1521.0 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   1521.0 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   1521.0 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   1521.0 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   1521.0 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   1521.0 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   1617.3 MiB     96.2 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   1521.0 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   1617.3 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   1617.3 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== CPU COMPUTATION, NO VRAM USAGE ========
    
    
     
    
    ========== VisualBERT_torchray_gpu.mlog ==========
    
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         8    463.6 MiB    463.6 MiB           1   def predict(model_type,
         9                                                     exp_method,
        10                                                     user_model="no_model",
        11                                                     img_name="profiling.png",
        12                                                     img_text="kill the jews and muslims for being anti-white",
        13                                                     model_path=None,
        14                                                     exp_direction="encourage"):
        15                                         
        16   2857.7 MiB   2394.1 MiB           2       model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        17    463.6 MiB      0.0 MiB           1           img_name, img_text, user_model, model_type, model_path, exp_direction)
        18                                         
        19   2857.7 MiB      0.0 MiB           1       print(type(model))
        20                                         
        21   2857.7 MiB      0.0 MiB           1       hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
        22                                         
        23   2857.7 MiB      0.0 MiB           2       cls_result = 'Your uploaded image and text combination ' \
        24                                                          'looks like a {} meme, with {}% confidence.'.format(
        25   2857.7 MiB      0.0 MiB           1                        hateful, "%.2f" % (cls_confidence * 100))
        26                                         
        27   2857.7 MiB      0.0 MiB           1       print(cls_result)
        28                                         
        29   2857.7 MiB      0.0 MiB           1       if exp_method == 'shap':
        30                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        31                                                     img_name, img_text, model, label_to_explain, model_output(cls_label, cls_confidence))
        32   2857.7 MiB      0.0 MiB           1       elif exp_method == 'lime':
        33                                                 text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
        34                                                     img_name, img_text, model, label_to_explain)
        35   2857.7 MiB      0.0 MiB           1       elif exp_method == 'torchray':
        36   3191.5 MiB    333.9 MiB           2           text_exp, img_exp, txt_msg, img_msg = torchray_mmf.torchray_multimodal_explain(
        37   2857.7 MiB      0.0 MiB           1               img_name, img_text, model, label_to_explain)
        38                                             else:
        39                                                 text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
        40                                                     img_name, img_text, model, label_to_explain)
        41                                         
        42   3191.5 MiB      0.0 MiB           1       img_exp_name, _ = os.path.splitext(img_exp)
        43   3191.5 MiB      0.0 MiB           1       exp_text_visl = img_exp_name + '_text.png'
    
    
    ======== VRAM INFO ========
    cuda:0 reserved 2992635904 bytes
    
     
    
    ========== inpainting.mlog ==========
    Filename: inpainting_time.py
    
    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
         5    108.8 MiB    108.8 MiB           1   def inpaint():
         6    217.6 MiB    108.8 MiB           1       remover = SmartTextRemover("../../mmxai/text_removal/frozen_east_text_detection.pb")
         7    336.7 MiB    119.1 MiB           1       img = remover.inpaint("https://www.iqmetrix.com/hubfs/Meme%2021.jpg")
