
import numpy as np
from mmxai.interpretability.classification.base_explainer import BaseExplainer

import warnings
from torchray.attribution.extremal_perturbation import extremal_perturbation
from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import multi_extremal_perturbation, text_extremal_perturbation
from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import image2tensor,PIL2tensor
from torchray.attribution.extremal_perturbation import (
    MaskGenerator,
    Perturbation,
    simple_reward,
    contrastive_reward,
)

BLUR_PERTURBATION = "blur"
"""Blur-type perturbation for :class:`Perturbation`."""

FADE_PERTURBATION = "fade"
"""Fade-type perturbation for :class:`Perturbation`."""

PRESERVE_VARIANT = "preserve"
"""Preservation game for :func:`extremal_perturbation`."""

DELETE_VARIANT = "delete"
"""Deletion game for :func:`extremal_perturbation`."""
# -------------------------- Explainer Object-----------------------------------------
class TorchRayExplainer(BaseExplainer):

    def __init__(self,
        model, 
        exp_method,
        areas=[0.1],
        perturbation=BLUR_PERTURBATION,
        max_iter=800,
        num_levels=8,
        step=7,
        sigma=21,
        jitter=True,
        variant=PRESERVE_VARIANT,
        print_iter=None,
        debug=False,
        reward_func=simple_reward,
        resize=False,
        resize_mode="bilinear",
        smooth=0,
        text_explanation_plot=False):
        perturbation_methods = (BLUR_PERTURBATION,FADE_PERTURBATION,PRESERVE_VARIANT,DELETE_VARIANT)
        reward_funcs = (simple_reward,contrastive_reward)
        for i in areas:
            if i < 0 or i > 1:
                raise ValueError(f"The preserved area have to be between 0 and 1")

        if perturbation not in perturbation_methods:
            raise ValueError(f"This perturbation method {perturbation} is not supported!")

        if variant not in perturbation_methods:
            raise ValueError(f"This variant method {variant} is not supported!")
        if reward_func not in reward_funcs:
            raise ValueError(f"This reward function {reward_func} is not supported!")
        super().__init__(model,exp_method)


        self.model = model
        self.areas=areas
        self.perturbation=perturbation
        self.max_iter=max_iter
        self.num_levels=num_levels
        self.step=step
        self.sigma=sigma
        self.jitter=jitter
        self.variant=variant
        self.print_iter=print_iter
        self.debug=debug
        self.reward_func=reward_func
        self.resize=resize
        self.resize_mode=resize_mode
        self.smooth=smooth
        self.text_explanation_plot=text_explanation_plot


    def explain(self, 
        image = None,
        text = None,
        target = 0
        ):
        self._mode = None
        image, text = self._parse_inputs(image,text)

        result = None
        if self._mode == "multimodal" or self._mode == "image_only":
            result = multi_extremal_perturbation(
                self.model,
                image,
                text[0],
                target,
                self.areas,
                self.perturbation,
                self.max_iter,
                self.num_levels,
                self.step,
                self.sigma,
                self.jitter,
                self.variant,
                self.print_iter,
                self.debug,
                self.reward_func,
                self.resize,
                self.resize_mode,
                self.smooth,
                self.text_explanation_plot
                )

        elif self._mode == "text_only":
            result = text_extremal_perturbation(self.model,
                text[0],
                target,
                self.text_explanation_plot
                )

        return result

    def _parse_inputs(self, image=None, text=None):
        """return compliant inputs for downstream
                Args:
                        as in .explain

        Returns:
                        (image, text)
                                - image: np.array of dimension (1, C, D1, D2)
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

                image = image2tensor(image)
            elif isinstance(image, np.ndarray):
                if image.shape[-1] != 3:
                    raise ValueError("Image arrays should have 3 channels.")
                if len(image.shape) == 4:
                    warnings.warn("Note: currently only N=1 is supported for image input.")
                    image = image[:1, ...]  # set N = 1
                    image = image.reshape(1,image.shape[-1],image.shape[1],image.shape[2])
                elif len(image.shape) == 3:
                    image = image[np.newaxis, ...]
                    image = image.reshape(1,image.shape[-1],image.shape[1],image.shape[2])
                else:
                    raise ValueError("Image array must of 3 or 4 dimensions.")

                image = torch.from_numpy(image)
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



if __name__ == "__main__":
    from mmf.models.mmbt import MMBT
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    dummy_explainer = TorchRayExplainer(model,"torchray")
    #.img = np.zeros((1,224,224,3),dtype = np.dtype('d'))
    dummy_explainer.explain(image="/Users/louitech_zero/Desktop/Imperial College London/CS/GroupProject/mmxai/mmxai/interpretability/classification/torchray/extremal_perturbation/test_img.jpeg" ,text="hello")
    print("success")
