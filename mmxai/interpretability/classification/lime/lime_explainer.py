

class LimeExplainer(BaseExplainer):

	def __init__(self, exp_method, model, image, text, label_to_exp, **kwargs):
		super().__init__(exp_method, model, image, text, label_to_exp, **kwargs)
