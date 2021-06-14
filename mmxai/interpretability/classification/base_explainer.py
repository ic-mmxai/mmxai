from PIL import Image
import numpy as np
from typing import TYPE_CHECKING, Optional, Union
import warnings
if TYPE_CHECKING:
	from .lime import LimeExplainer
	from .shap import Explainer as ShapExplainer


class BaseExplainer(object):
	"""Interface for explainer methods"""

	def __init__(self, model, exp_method: str, **kwargs):
		"""Init function

        Args:
			exp_method: {"lime", "shap", "torchray"}
            model: ML predictor
        """

		self.model = model
		self.explainer_params = kwargs
		# "multi" for multimodal or "single" for singlemodal inputs
		self._mode = "multi"

		if self.__class__ is BaseExplainer:
			if exp_method == "lime":
				self.__class__ = explainers.LimeExplainer
				explainers.LimeExplainer.__init__(self, self.model, exp_method, **kwargs)
			elif exp_method == "shap":
				self.__class__ = ShapExplainer
				ShapExplainer.__init__(self, **kwargs)


	def explain(self, image:Optional[Union[np.ndarray, str]]=None, text:Optional[Union[np.ndarray, str]]=None, label_to_exp: Optional[int] = 0, **kwargs):
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
		self.mode = None
		assert False, f"Cannot call virtual function 'explain' from a {self.__class__} object, please implement this function."

	def _parse_inputs(self, image=None, text=None):
		""" return compliant inputs for downstream
		Args:
			as in .explain

        Returns:
			(image, text)
				- image: np.array of dimension (1, D1, D2, C)
				- text: np.array of dimension (1,)
		"""
		if image is not None and text is not None:
			self._mode = "multi"
		elif image is None : # got text only
			self._mode = "single"
		elif text is None: # got image only
			self._mode = "single"
		else: 
			raise ValueError("Both inputs cannot be None at the same time")

		if image:
			if isinstance(image, str):
				im = Image.open("image.png").convert('RGB')
				image = np.array(im)[np.newaxis, ...]
			if isinstance(image, np.ndarray):
				if image.shape[-1] != 3:
					raise ValueError("Image arrays should have 3 channels.")
				if len(image.shape) == 4:
					warnings("Note: currently only N=1 is supported for image input.")
					image = image[:1, ...] # set N = 1
				elif len(image.shape) == 3:
					image = image[np.newaxis, ...]
				else:
					raise ValueError("Image array must of 3 or 4 dimensions.")
			else:
				raise ValueError("Unknown iamge input passed in.")

		if text:
			if isinstance(text, str):
				text = np.array([text])
			elif isinstance(text, np.ndarray):
				if len(text.shape) > 1:
					raise ValueError("Text array must be of 1 dimension.")
				text = np.array(text)[:1] # set N = 1
			else:
				raise ValueError("Unknown text input passed in.")

		return image, text
