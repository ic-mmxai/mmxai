from mmxai.utils.image_loader import loadImage
from base_explanation_result import ExplanationResult
from lime.lime_explainer import LimeExplainer


class BaseExplainer(object):
	"""Interface for explainer methods"""

	def __init__(self, exp_method, model, image, text, label_to_exp, **kwargs):
		"""Init function

        Args:
            model: ML predictor
            image: image path or numpy array or PIL Image object
            text: string
            label_to_exp: label name
        """

		self.model = model
		self.image = loadImage(image)
		self.text = text
		self.label = label_to_exp
		self.explainer_params = kwargs

		if self.__class__ is BaseExplainer:
			if exp_method == "lime":
            	self.__class__ = explainers.LimeExplainer
            	explainers.LimeExplainer.__init__(self, self.exp_method, self.model, self.image, self.text, self.label, **kwargs)


	def explain():
		"""Init function

        Return:
			res: Dict{img_img=None, txt_img=None, img_exp=None, txt_exp=None, ...}
		"""
		assert False

