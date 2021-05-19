import onnxruntime
import numpy as np
import torch
from torch import nn
import onnx
from onnx import helper, TensorProto, checker

from torchvision import transforms
from torchray.utils import imsc
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer, BertTokenizerFast, XLNetTokenizer
from mmf.models.mmbt import MMBT

class ONNXInterface:
	def __init__(self,model_path,tokenizer = None):

		'''
			Initilize interface by rebuild model from model path and tokenizer
		'''
		self.model = onnx.load(model_path)
		self.ort_session = onnxruntime.InferenceSession(model_path)
		if not onnx.checker.check_model(self.model):
			assert("Model file error")
		
		 
		self.defaultmodel = None
		self.device = "cpu"

		if tokenizer != None:
			if tokenizer == "BertTokenizer":
				self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
			elif tokenizer == "BertTokenizerFast":
				self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
			elif tokenizer == "AutoTokenizer":
				self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
			elif tokenizer == "XLNetTokenizer":
				self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
			else:
				assert("NotImplementedError")
				print("Please contact the development team to update")

	def visualize(self):
		'''
			visualize model structure
		'''
		print(onnx.helper.printable_graph(self.model.graph))

	def onnx_model_forward(self, image_input,text_input):
		'''
		It is an model oriented function which will supports several models with different Input Type
		Args:
			image_input: the image torch.tensor with size (1,3,224,224)
			text_input : the text input Str
		Returns :
			logits computed by model.forward List()
		'''
		output_name = self.ort_session.get_outputs()[0].name
		inputs = []
		count = 0
		while True:
			try:
				input_name = self.ort_session.get_inputs()[count].name
				inputs.append(input_name)
				count += 1
			except:
				if count > 4:
					print("The input model is not bert or MMF models, they are not supported please contact the development teams")
					exit(0)
				break
		
		img = to_numpy(image_input)
		Tokenizer = self.tokenizer
		if count == 3:
			tokens = Tokenizer(text_input, return_tensors="pt")
			ort_inputs = {k: v.cpu().detach().numpy() for k, v in tokens.items()}

		elif count == 2:

			input1 = Tokenizer(text_input, return_tensors="pt")["input_ids"].squeeze().type(torch.float)
			ort_inputs = {inputs[0]: img, inputs[1]: input1}

		elif count == 4:

			input1 = Tokenizer(text_input, return_tensors="pt")["input_ids"].cpu().detach().numpy()
			input2 = Tokenizer(text_input, return_tensors="pt")["token_type_ids"].cpu().detach().numpy()
			input3 = Tokenizer(text_input, return_tensors="pt")["attention_mask"].cpu().detach().numpy()
			ort_inputs = {inputs[0]: img, inputs[1]: input1,inputs[2]: input2,inputs[3]: input3}
			 
		else:

			ort_inputs = {inputs[0] : img}


		ort_outs = self.ort_session.run([output_name], ort_inputs)
		return ort_outs

	def to(self,device):
		self.device = device
		
	def classify(self,image,text_input, image_tensor = None):
		'''
		Args:	
			image_path: directory of input image
			text_input : the text input Str
			image_tensor : the image torch.tensor with size (1,3,224,224)
			
		Returns :
			label of model prediction and the corresponding confidence
		'''
		
		scoreFlag = False
		if image_tensor != None:
			scoreFlag = True
			logits = self.onnx_model_forward(image_tensor,text_input)
		else:
			p = transforms.Compose([transforms.Scale((224,224))])
			image,i = imsc(p(image),quiet=True)
			image_tensor = torch.reshape(image, (1,3,224,224))
			logits = self.onnx_model_forward(image_tensor,text_input)

		if list(torch.tensor(logits).size()) != [1, 2]:
			
			if self.defaultmodel == None:
				self.defaultmodel = MMBT.from_pretrained("mmbt.hateful_memes.images")
				self.defaultmodel.to(self.device)
			logits = self.defaultmodel.classify(image, text_input, image_tensor=torch.squeeze(image_tensor.to(self.device), 0))
			

		scores = nn.functional.softmax(torch.tensor(logits), dim=1)

		if scoreFlag == True:
			return scores

		confidence, label = torch.max(scores, dim=1)

		return {"label": label.item(), "confidence": confidence.item()}


def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224,224))])

    img,i = imsc(p(img),quiet=True)
    return torch.reshape(img, (1,3,224,224))



def to_numpy(tensor):
	"""
		convert torch tensor to numpy array
	"""
	return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
	model_path = "Bert.onnx"
	tokenizers = ["BertTokenizer","BertTokenizerFast","AutoTokenizer","XLNetTokenizer"]
	tokenizer = tokenizers[0]
	model = ONNXInterface(model_path,tokenizer)

