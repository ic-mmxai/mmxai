# Copyright (c) Facebook, Inc. and its affiliates.

# Used for MMF internal models for hateful memes task that make predictions
# on raw images

import os
import tempfile
from pathlib import Path
from typing import Type, Union
import numpy as np
import torch
import torchvision.datasets.folder as tv_helpers
from omegaconf import DictConfig
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_processors
from mmf.utils.download import download
from PIL import Image
from torch import nn

from tools.scripts.features.frcnn.extract_features_frcnn import FeatureExtractor

ImageType = Union[Type[Image.Image], str]
PathType = Union[Type[Path], str]
BaseModelType = Type[BaseModel]
from torchray_visual_interface import * 

class FeatureModelInterface(nn.Module):
    def __init__(self, model: BaseModelType, config: DictConfig, model_name: str):
        super().__init__()
        self.model = model
        self.config = config
        self.processor_dict = None
        self.model_name = model_name
        self.init_processors()
        self.feature_extractor = FeatureExtractor()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_processors(self):
        config = self.config.dataset_config.hateful_memes
        extra_params = {"data_dir": config.data_dir}
        self.processor_dict = build_processors(config.processors, **extra_params)

    def __call__(self, image: ImageType, text: str, image_tensor = None,zero_image=False, zero_text=False):
        """Classifies a given image and text in it into Hateful/Non-Hateful.
        Image can be a url or a local path or you can directly pass a PIL.Image.Image
        object. Text needs to be a sentence containing all text in the image.

        Args:
            image (ImageType): Image to be classified
            text (str): Text in the image
            zero_image: zero out the image features when classifying
            zero_text: zero out the text features when classifying

        Returns:
            {"label": 0, "confidence": 0.56}
        """

        if image_tensor != None:
            image_tenosr = torch.unsqueeze(image_tenosr,0)
            im_feature_0, im_info_0 = torchRay_feat_extract(image_tensor)
        else:
            if isinstance(image, str):
                if image.startswith("http"):
                    temp_file = tempfile.NamedTemporaryFile()
                    download(image, *os.path.split(temp_file.name), disable_tqdm=True)
                    image = tv_helpers.default_loader(temp_file.name)
                    temp_file.close()
                else:
                    image = tv_helpers.default_loader(image)
            _, _, im_feature_0, im_info_0 = self.feature_extractor.extract_features(
                image_dir=image, save_single=False
            )

            # expected case, only support passing one instance
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    image = Image.fromarray(image.astype('uint8'), 'RGB')

                elif len(image.shape) == 4:
                    if image.shape[0] == 1:
                        image = image[0, :]
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                    else:
                        raise ValueError("This model currently cannot process multiple inputs.")
                        return None
                else: 
                    raise ValueError("Image input dimension is not correct.")
                    return None

            else:
                ValueError("Only type str and numpy.ndarray are supported")
                return None


        if isinstance(text, str):
            text = self.processor_dict["text_processor"]({"text": text})
        elif isinstance(text, list) or isinstance(text, np.ndarray):
            if len(text) == 1:
                text = self.processor_dict["text_processor"]({"text": text[0]})
            else:
                raise ValueError("This model currently cannot process multiple inputs.")
                return None
        else:
            raise ValueError("Only type str and list are supported")
            return None              

        sample = Sample()
        sample.text = text["text"]
        if "input_ids" in text:
            sample.update(text)

        # extract feature
        #_, _, im_feature_0, im_info_0 = self.feature_extractor.extract_features(
        #    image_dir=image, save_single=False
        #)

        # re-format the sample list
        sample_im_info = Sample()

        # process the bounding boxes for vilbert
        if self.model_name == "vilbert":
            bbox = np.array(im_info_0["bbox"])
            image_w = im_info_0["image_width"]
            image_h = im_info_0["image_height"]
            new_bbox = np.zeros((bbox.shape[0], 5), dtype=bbox.dtype)

            new_bbox[:, 0] = bbox[:, 0] / image_w
            new_bbox[:, 1] = bbox[:, 1] / image_h
            new_bbox[:, 2] = (bbox[:, 2]) / image_w
            new_bbox[:, 3] = (bbox[:, 3]) / image_h
            new_bbox[:, 4] = (
                (bbox[:, 2] - bbox[:, 0])
                * (bbox[:, 3] - bbox[:, 1])
                / (image_w * image_h)
            )

            sample_im_info.bbox = torch.from_numpy(new_bbox)
        else:
            sample_im_info.bbox = torch.from_numpy(np.array(im_info_0["bbox"]))

        sample_im_info.num_boxes = torch.from_numpy(np.array(im_info_0["num_boxes"]))
        sample_im_info.objects = torch.from_numpy(np.array(im_info_0["objects"]))
        sample_im_info.image_width = torch.from_numpy(
            np.array(im_info_0["image_width"])
        )
        sample_im_info.image_height = torch.from_numpy(
            np.array(im_info_0["image_height"])
        )
        sample_im_info.cls_prob = torch.from_numpy(np.array(im_info_0["cls_prob"]))
        sample_list_info = SampleList([sample_im_info])

        sample.image_feature_0 = im_feature_0
        sample.dataset_name = "hateful_memes"

        sample_list = SampleList([sample])
        sample_list.image_info_0 = sample_list_info
        device = next(self.model.parameters()).device
        sample_list = sample_list.to(device)

        output = self.model(sample_list)
        scores = nn.functional.softmax(output["scores"], dim=1)

        if image_tensor != None:
            return scores
        else:
            return scores.detach().numpy()

            