from mmxai.utils.image_loader import loadImage
from base_explanation_result import ExplanationResult
from lime.lime_explainer import LimeExplainer
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

		if self.__class__ is BaseExplainer:
			if exp_method == "lime":
            	self.__class__ = explainers.LimeExplainer
            	explainers.LimeExplainer.__init__(self, self.exp_method, self.model, self.image, self.text, self.label, **kwargs)
			elif exp_method == "shap":
				self.__class__ = ShapExplainer
				ShapExplainer.__init__(self, **kwargs)




	def explain(self, image, text, label_to_exp, **kwargs):
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

