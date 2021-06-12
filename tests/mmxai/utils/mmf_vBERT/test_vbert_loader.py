from mmxai.utils.mmf_vBERT.vbert_loader import loadPretrainedVisualBERT, loadPretrainedVilBERT

import os, shutil

def testCanLoadVilBERT():
    dir_path = os.path.expanduser(
                "~/.cache/torch/mmf/data/models/vilbert.finetuned.hateful_memes.from_cc_original")
    
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    loadPretrainedVilBERT()


def testCanLoadVisualBERT():
    dir_path = os.path.expanduser(
                "~/.cache/torch/mmf/data/models/visual_bert.finetuned.hateful_memes.from_coco")
    
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    loadPretrainedVisualBERT()

if __name__ == "__main__":
    testCanLoadVisualBERT()
    testCanLoadVilBERT()