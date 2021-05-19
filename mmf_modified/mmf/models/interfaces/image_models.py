# Copyright (c) Facebook, Inc. and its affiliates.

# Used for MMF internal models, hateful memes task,
# make predictions on raw images and texts

import os
import tempfile
from pathlib import Path
from typing import Type, Union

import torch
import torchvision.datasets.folder as tv_helpers
from omegaconf import DictConfig
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_processors
from mmf.utils.download import download
from PIL import Image
from torch import nn


ImageType = Union[Type[Image.Image], str]
PathType = Union[Type[Path], str]
BaseModelType = Type[BaseModel]


class GeneralInterface(nn.Module):
    def __init__(self, model: BaseModelType, config: DictConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.init_processors()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_processors(self):
        config = self.config.dataset_config.hateful_memes
        extra_params = {"data_dir": config.data_dir}
        self.processor_dict = build_processors(config.processors, **extra_params)

    def classify(
        self,
        image: ImageType,
        text: str,
        image_tensor=None,
        zero_image=False,
        zero_text=False,
    ):
        """Classifies a given image and text in it into Hateful/Non-Hateful.
        Image can be a url or a local path or you can directly pass a PIL.Image.Image
        object. Text needs to be a sentence containing all text in the image.

        Args:
            image (ImageType): Image to be classified
            text (str): Text in the image
            zero_image: zero out the image features when classifying
            zero_text: zero out the text features when classifying
            return_type: either "prob" or "logits"

        Returns:
            {"label": 0, "confidence": 0.56}
        """
        sample = Sample()

        if image_tensor != None:
            #if len(image_tensor.shape) != 4:
            #    image_tensor = torch.unsqueeze(image_tensor,0)
            sample.image = image_tensor
        else:

            if isinstance(image, str):
                if image.startswith("http"):
                    temp_file = tempfile.NamedTemporaryFile()
                    download(image, *os.path.split(temp_file.name), disable_tqdm=True)
                    image = tv_helpers.default_loader(temp_file.name)
                    temp_file.close()
                else:
                    image = tv_helpers.default_loader(image)

            image = self.processor_dict["image_processor"](image)
            sample.image = image

        text = self.processor_dict["text_processor"]({"text": text})

        sample.text = text["text"]
        if "input_ids" in text:
            sample.update(text)

        sample_list = SampleList([sample])
        device = next(self.model.parameters()).device
        sample_list = sample_list.to(device)
        output = self.model(sample_list, zero_image=zero_image, zero_text=zero_text)
        scores = nn.functional.softmax(output["scores"], dim=1)

        if image_tensor != None:
            return scores

        confidence, label = torch.max(scores, dim=1)

        return {"label": label.item(), "confidence": confidence.item()}
