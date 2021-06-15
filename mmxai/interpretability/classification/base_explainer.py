from PIL import Image
import numpy as np
from typing import TYPE_CHECKING, Optional, Union
import warnings
from .. import classification


class BaseExplainer(object):
    """Interface for explainer methods"""

    __supported_modes = (
        "image_only",
        "multimodal",
    )

    __supported_methods = ("lime", "shap", "torchray")

    def __init__(self, model, exp_method: str, **kwargs):
        """Init function

        Args:
            exp_method: {"lime", "shap", "torchray"}
            model: ML predictor
                it should take in input dataset(s) each of dimension (N, ...) 
                where N is the number of samples in the datase.
                its output should be of dimension (N, num_labels) of probabilities.
        """

        self.model = model

        # model should be callable
        if not hasattr(model, "__call__"):
            raise ValueError(f"Model passed in must be callable.")

        self.explainer_params = kwargs
        self._mode = None

        print(self.__class__)
        if self.__class__ is BaseExplainer:
            if exp_method == "lime":
                self.__class__ = classification.LimeExplainer
                classification.LimeExplainer.__init__(
                    self, self.model, exp_method, **kwargs
                )
            elif exp_method == "shap":
                self.__class__ = classification.ShapExplainer
                classification.ShapExplainer.__init__(self, self.model, **kwargs)

    def explain(
        self,
        image: Optional[Union[np.ndarray, str]] = None,
        text: Optional[Union[np.ndarray, str]] = None,
        label_to_exp: Optional[int] = 0,
        **kwargs,
    ):
        """Init function
                Args:
            image: image path (str) or numpy array of dimension (N, D1, D2, C)
                                - N = number of samples
                                - D1, D2, C are dimensions of an image array, C = 3
            text: str or array/list of strings where len(texts) = N as above
            label_to_exp: label index to explain, default = 0
                        NOTE: only supports N=1 is supported for now
                                so it is possible to pass in an array of dimension (D1, D2, C) and a string for text

        Returns:
                        res: Dict{img_img=None, txt_img=None, img_exp=None, txt_exp=None, ...}
        """
        self.image = loadImage(image)
        self.text = text
        self.label = label_to_exp
        self._mode = None
        assert (
            False
        ), f"Cannot call virtual function 'explain' from a {self.__class__} object, please implement this function."

    def _parse_inputs(self, image=None, text=None):
        """return compliant inputs for downstream
                Args:
                        as in .explain

        Returns:
                        (image, text)
                                - image: np.array of dimension (1, D1, D2, C)
                                - text: np.array of dimension (1,)
        """
        if image is not None and text is not None:
            self._mode = "multimodal"
        elif image is None:  # got text only
            self._mode = "text_only"
        elif text is None:  # got image only
            self._mode = "image_only"
        else:
            raise ValueError("Both inputs cannot be None at the same time")

        if image is not None:
            if isinstance(image, str):
                im = Image.open("image.png").convert("RGB")
                image = np.array(im)[np.newaxis, ...]
            if isinstance(image, np.ndarray):
                if image.shape[-1] != 3:
                    raise ValueError("Image arrays should have 3 channels.")
                if len(image.shape) == 4:
                    warnings.warn("Note: currently only N=1 is supported for image input.")
                    image = image[:1, ...]  # set N = 1
                elif len(image.shape) == 3:
                    image = image[np.newaxis, ...]
                else:
                    raise ValueError("Image array must of 3 or 4 dimensions.")
            else:
                raise ValueError("Unknown iamge input passed in.")

        if text is not None:
            if isinstance(text, str):
                text = np.array([text])
            elif isinstance(text, np.ndarray):
                if len(text.shape) > 1:
                    raise ValueError("Text array must be of 1 dimension.")
                text = np.array(text)[:1]  # set N = 1
            else:
                raise ValueError("Unknown text input passed in.")

        return image, text
