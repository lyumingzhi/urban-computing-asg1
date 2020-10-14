import torch.nn as nn
import torch.nn.functional as F
import torch
class MLP(nn.Module):
	def __init__(inputsize,h1size,h2size,outputsize):
		super(MLP,self).__init__()
		self.layer1=nn.Linear(inputsize,h1size)
		self.layer2=nn.Linear(h1size,h2size)
		self.layer3=nn.Linear(h2size,outputsize)
	def forward(x):
		x=self.layer1(x)
		x=self.layer2(x)
		y=self.layer3(x)
		return y
