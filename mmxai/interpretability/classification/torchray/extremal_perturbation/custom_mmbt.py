from mmf.models.interfaces.mmbt import MMBTGridHMInterface

import os
import tempfile
import torchvision.datasets.folder as tv_helpers
from torch import nn

from mmf.utils.download import download
from mmf.common.sample import Sample, SampleList
from mmf.utils.general import *


class MMBTGridHMInterfaceOnlyImage(MMBTGridHMInterface):
    """Interface for MMBT Grid for Hateful Memes which only require image input."""

    def __init__(self, model: MMBTGridHMInterface, text: str):
        super().__init__(model.model, model.config)
        self.__text = text

    @property
    def text(self):
        return self.__text

    def classify(self, image, text):

        return super().classify(image, text)

    def imageToTensor(self, image):
        """
        Transform the input image into tensor form.
        This function "take out" one of the sub-step in super().classify()
        """
        if isinstance(image, str):
            if image.startswith("http"):
                temp_file = tempfile.NamedTemporaryFile()
                download(image, *os.path.split(temp_file.name), disable_tqdm=True)
                image = tv_helpers.default_loader(temp_file.name)
                temp_file.close()
            else:
                image = tv_helpers.default_loader(image)

        return self.processor_dict["image_processor"](image)

    def __call__(self, image_tensor, text_input=None):
        """
        Allow model to receive both multi-inputs and single image-inputs // Bojia Mao
        """
        text = self.processor_dict["text_processor"]({"text": self.text})

        sample = Sample()

        if text_input == None:
            sample.text = text["text"]
        else:
            self.__text = text_input
            sample.text = text_input

        if "input_ids" in text:
            sample.update(text)

        sample.image = image_tensor
        sample_list = SampleList([sample])
        sample_list = sample_list.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        output = self.model(sample_list)
        scores = nn.functional.softmax(output["scores"], dim=1)

        return scores


def main():  # pragma: no cover
    import matplotlib.pyplot as plt

    # import requests
    import torch

    from PIL import Image

    from mmf.models.mmbt import MMBT

    # Check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = "./hateful_memes/example.jpg"
    image = Image.open(image_path)
    text = "look how many people love you"

    model = MMBTGridHMInterfaceOnlyImage(
        MMBT.from_pretrained("mmbt.hateful_memes.images"), text
    )
    model.to(device)  # Move model to GPU if cuda is available

    output = model.classify(image)

    plt.imshow(image)
    plt.axis("off")
    plt.show()

    hateful = "Yes" if output["label"] == 1 else "No"

    print("Hateful as per the model?", hateful)
    print(f"Model's confidence: {output['confidence'] * 100:.3f}%")


if __name__ == "__main__":  # pragma: no cover
    pass
