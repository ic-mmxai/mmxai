from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT

import os

def loadPretrainedVilBERT():
    error_count = 1

    try:
        return ViLBERT.from_pretrained(
            "vilbert.finetuned.hateful_memes.from_cc_original")
    except:
        if error_count > 0:
            config_path = os.path.expanduser(
                "~/.cache/torch/mmf/data/models/vilbert.finetuned.hateful_memes.from_cc_original/config.yaml")
            fixConfig(config_path)
            error_count -= 1
        else:
            raise

def loadPretrainedVisualBERT():
    error_count = 1

    try:
        return VisualBERT.from_pretrained(
            "visual_bert.finetuned.hateful_memes.from_coco")
    except:
        if error_count > 0:
            config_path = os.path.expanduser(
                "~/.cache/torch/mmf/data/models/visual_bert.finetuned.hateful_memes.from_coco/config.yaml")
            fixConfig(config_path)
            error_count -= 1
        else:
            raise


def fixConfig(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    for n, line in enumerate(lines):
        if line.startswith("  cache_dir:"):
            lines[n] = "  cache_dir: ${resolve_cache_dir:MMF_CACHE_DIR}\n"
        if line.startswith("    data_dir:"):
            lines[n] = "    data_dir: ${resolve_dir:MMF_DATA_DIR, data}\n"
        if line.startswith("  data_dir:"):
            lines[n] = "  data_dir: ${resolve_dir:MMF_DATA_DIR, data}\n"
        if line.startswith("  save_dir:"):
            lines[n] = "  save_dir: ${env:MMF_SAVE_DIR, ./save}\n"
        if line.startswith("  resume_file:"):
            lines[n] = "  resume_file: null\n"

    with open(path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    loadPretrainedVilBERT()
    loadPretrainedVisualBERT()