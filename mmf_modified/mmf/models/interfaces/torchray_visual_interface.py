import numpy as np
import torch
from PIL import Image
import argparse


from torch import nn
from torchvision import transforms
from torchray.utils import imsc


from tools.scripts.features.frcnn.modeling_frcnn import GeneralizedRCNN
from tools.scripts.features.frcnn.frcnn_utils import Config





def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="LXMERT",
        type=str,
        help="Model to use for detection",
    )
    parser.add_argument(
        "--model_file",
        default=None,
        type=str,
        help="Huggingface model file. This overrides the model_name param.",
    )
    parser.add_argument(
        "--config_file", default=None, type=str, help="Huggingface config file"
    )
    parser.add_argument(
        "--start_index", default=0, type=int, help="Index to start from "
    )
    parser.add_argument("--end_index", default=None, type=int, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_features",
        type=int,
        default=100,
        help="Number of features to extract.",
    )
    parser.add_argument(
        "--output_folder", type=str, default="./output", help="Output folder"
    )
    parser.add_argument("--image_dir", type=str, help="Image directory or file")
    # TODO add functionality for this flag
    parser.add_argument(
        "--feature_name",
        type=str,
        help="The name of the feature to extract",
        default="fc6",
    )
    parser.add_argument(
        "--exclude_list",
        type=str,
        help="List of images to be excluded from feature conversion. "
        + "Each image on a new line",
        default="./list",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0,
        help="Threshold of detection confidence above which boxes will be selected",
    )
    # TODO finish background flag
    parser.add_argument(
        "--background",
        action="store_true",
        help="The model will output predictions for the background class when set",
    )
    parser.add_argument(
        "--padding",
        type=str,
        default=None,
        help="You can set your padding, i.e. 'max_detections'",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="Add this flag to save the extra file used for visualization",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Add this flag to save the extra file used for visualization",
    )
    parser.add_argument(
        "--max_partition",
        type=int,
        default=None,
        help="Add this flag to save the extra file used for visualization",
    )
    return parser


def process_features(features, index, args):
    feature_keys = [
        "obj_ids",
        "obj_probs",
        "attr_ids",
        "attr_probs",
        "boxes",
        "sizes",
        "preds_per_image",
        "roi_features",
        "normalized_boxes",
    ]
    single_features = dict()

    for key in feature_keys:
        single_features[key] = features[key][index]

    confidence = args.confidence_threshold
    idx = 0
    while idx < single_features["obj_ids"].size()[0]:
        removed = False
        if (
            single_features["obj_probs"][idx] < confidence
            or single_features["attr_probs"][idx] < confidence
        ):
            single_features["obj_ids"] = torch.cat(
                [
                    single_features["obj_ids"][0:idx],
                    single_features["obj_ids"][idx + 1 :],
                ]
            )
            single_features["obj_probs"] = torch.cat(
                [
                    single_features["obj_probs"][0:idx],
                    single_features["obj_probs"][idx + 1 :],
                ]
            )
            single_features["attr_ids"] = torch.cat(
                [
                    single_features["attr_ids"][0:idx],
                    single_features["attr_ids"][idx + 1 :],
                ]
            )
            single_features["attr_probs"] = torch.cat(
                [
                    single_features["attr_probs"][0:idx],
                    single_features["attr_probs"][idx + 1 :],
                ]
            )
            single_features["boxes"] = torch.cat(
                [
                    single_features["boxes"][0:idx, :],
                    single_features["boxes"][idx + 1 :, :],
                ]
            )
            single_features["preds_per_image"] = single_features["preds_per_image"] - 1
            single_features["roi_features"] = torch.cat(
                [
                    single_features["roi_features"][0:idx, :],
                    single_features["roi_features"][idx + 1 :, :],
                ]
            )
            single_features["normalized_boxes"] = torch.cat(
                [
                    single_features["normalized_boxes"][0:idx, :],
                    single_features["normalized_boxes"][idx + 1 :, :],
                ]
            )
            removed = True
        if not removed:
            idx += 1

    feat_list = single_features["roi_features"]

    boxes = single_features["boxes"][: args.num_features].cpu().numpy()
    num_boxes = args.num_features
    objects = single_features["obj_ids"][: args.num_features].cpu().numpy()
    probs = single_features["obj_probs"][: args.num_features].cpu().numpy()
    width = single_features["sizes"][1].item()
    height = single_features["sizes"][0].item()
    info_list = {
        "bbox": boxes,
        "num_boxes": num_boxes,
        "objects": objects,
        "cls_prob": probs,
        "image_width": width,
        "image_height": height,
    }

    return single_features, feat_list, info_list


# image to 4-D torch tensor
def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(img), quiet=True)
    return torch.reshape(img, (1, 3, 224, 224))


# extract image feature
def torchRay_feat_extract(img_tensor):
    Args = get_parser().parse_args(
        [
            "--config_file",
            "/Users/louitech_zero/Desktop/Imperial College London/CS/GroupProject/group_project_draft/mmf/tools/scripts/features/frcnn/config.yaml",
            "--model_file",
            "/Users/louitech_zero/Desktop/Imperial College London/CS/GroupProject/group_project_draft/mmf/tools/scripts/features/frcnn/model_finetuned.bin",
        ]
    )
    feature_extraction_model = GeneralizedRCNN.from_pretrained(
        Args.model_file, config=Config.from_pretrained(Args.config_file)
    )

    features = feature_extraction_model(
        img_tensor,
        torch.tensor([[224, 224]]),
        scales_yx=torch.tensor([[1.0, 1.0]]),
        padding=None,
        max_detections=Config.from_pretrained(Args.config_file).max_detections,
        return_tensors="pt",
    )
    single_features, feat_list, info_list = process_features(features, 0, Args)

    return feat_list, info_list
