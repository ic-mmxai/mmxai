from mmxai.utils.image_loader import loadImage
from typing import TYPE_CHECKING
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
		self.num_labels = 2 # would be good to get this somehow

		if self.__class__ is BaseExplainer:
			if exp_method == "lime":
				self.__class__ = explainers.LimeExplainer
				explainers.LimeExplainer.__init__(self, self.model, exp_method, **kwargs)
			elif exp_method == "shap":
				self.__class__ = ShapExplainer
				ShapExplainer.__init__(self, **kwargs)


	def explain(self, image=None, text:str = None, label_to_exp:int = 0, **kwargs):
		"""Init function
		Args:
            image: image path or numpy array or PIL Image object
            text: string
            label_to_exp: label name

        Returns:
			res: Dict{img_img=None, txt_img=None, img_exp=None, txt_exp=None, ...}
		"""
		self.image = loadImage(image)
		self.text = text
		self.label = label_to_exp
		assert False, f"Cannot call virtual function 'explain' from a {self.__class__} object, please implement this function."
