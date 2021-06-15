from mmxai.interpretability.classification.lime.lime_multimodal import LimeMultimodalExplainer, MultiModalExplanation
from mmxai.interpretability.classification.base_explainer import BaseExplainer
from mmxai.utils.image_loader import loadImage
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


class LimeExplainer(BaseExplainer):
	'''The wrapper class of extended lime explainer'''

	def __init__(self, model, exp_method, **kwargs):

		super().__init__(model, exp_method, **kwargs)


	def explain(self, image=None, text = None, label_to_exp = 0, **kwargs):

		self.image = image[0]
		self.text = text[0]
		self.label = label_to_exp
		if "num_samples" in kwargs:
			self.num_samples = kwargs[num_samples]
		else:
			self.num_samples = 500

		# define predict function to use in LIME
		'''def multi_predict(model, imgs, txts, zero_image=False, zero_text=False):
			inputs = zip(imgs, txts)
			res = np.zeros((len(imgs), 2))
			for i, this_input in enumerate(inputs):
				img = Image.fromarray(this_input[0])
				txt = this_input[1]
				try:
					this_output = model(
						img, txt, zero_image=zero_image, zero_text=zero_text
					)
				except:
					this_output = model(img, txt)
				res[i][this_output["label"]] = this_output["confidence"]
				res[i][1 - this_output["label"]] = 1 - this_output["confidence"]
			return res'''

		def multi_predict(model, imgs, txts, zero_image=False, zero_text=False):
			try:
				# the model can handle multiple inputs
				res = model(imgs, txts)
				return res
			except:
				# the model can only handle one input at a time
				num_labels =  model(imgs[0], txts[0]).shape[1]
				res = np.zeros((len(imgs), num_labels)) # place holder
				for i in range(len(imgs)):
					img = imgs[i]
					txt = txts[i]
					try:
						this_output = model(
							img, txt, zero_image=zero_image, zero_text=zero_text
						)
					except:
						this_output = model(img, txt)
					res[i] = this_output
				return res


	    # generate explanation
		exp1 = LimeMultimodalExplainer(self.image, self.text, self.model)
		explanation1 = exp1.explain_instance(multi_predict, self.num_samples)
		txt_exp, img_exp, text_list, temp, mask = explanation1.get_explanation(self.label)

	    # image explanation result visualisation
		img_boundry = mark_boundaries(temp, mask)
		img_boundry = img_as_ubyte(img_boundry)
		img_img = Image.fromarray(np.uint8(img_boundry)).convert("RGB")

		# text explanation visualisation
		text_exp_list = []
		for pair in text_list:
			text_exp_list.append(list(pair))

		def get_second(element):
			return element[1]
		text_exp_list = sorted(text_exp_list, key=get_second, reverse=True)
		txt_img = _text_visualisation(text_exp_list, self.label)

	    # result
		res = dict()
		res["img_img"] = img_img
		res["txt_img"] = txt_img
		res["txt_exp"] = txt_exp
		res["img_exp"] = img_exp
		return res


	def _text_visualisation(self, exp):
		if self.label == 1:
			plt_title = "hateful"
		else:
			plt_title = "not hateful"

		vals = []
		names = []
		if isinstance(exp, dict):
			for i in exp:
				names.append(i)
				vals.append(exp[i])
		elif isinstance(exp, list) and isinstance(exp[0], str):
			for i in exp:
				names.append(i.split()[0])
				vals.append(float(i.split()[-1]))
		elif isinstance(exp, list) and isinstance(exp[0], list):
			vals = [x[1] for x in exp]
			names = [x[0] for x in exp]

		vals.reverse()
		names.reverse()

		fig = plt.figure(figsize=(15, 0.75 + 0.375 * len(vals)))

		colors = ["hotpink" if x > 0 else "cornflowerblue" for x in vals]
		pos = np.arange(len(exp)) + 0.5
		plt.barh(pos, vals, align="center", color=colors)

		plt.yticks(pos, names, fontsize=25)
		plt.xticks(fontsize=15)

		# convert pyplot object to PIL.Image
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		im = Image.open(buf)

		return im

if __name__ == "__main__":
	from mmf.models.mmbt import MMBT
	model = MMBT.from_pretrained("mmbt.hateful_memes.images")
	dummy_explainer = LimeExplainer(model, "lime")
	print("success")
