# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		shape = (1, num_features)
		self.weight = nn.Parameter(torch.ones(shape))
		self.bias = nn.Parameter(torch.zeros(shape))

		# Store the average mean and variance
		self.rm = torch.ones(shape)
		self.rv = torch.zeros(shape)
		self.register_buffer('running_mean', self.rm)
		self.register_buffer('running_var', self.rv)
		
		# Initialize your parameter
		self.eps = 1e-5

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		input_mean = input.mean(axis=0)
		# print(input_mean.shape)
		input_var = input.var(axis=0)
		# print(input_var.shape)
		# print("rm",self.rm.shape,"innput", input_mean.shape)
		self.running_mean = 0.1 * self.running_mean + 0.9 * input_mean
		self.running_var = 0.1 * self.running_var + 0.9 * input_var
		input_hat = (input - self.rm) / torch.sqrt(self.rv + self.eps)
		# print(self.weight.shape, self.bias.shape)
		input = input_hat * self.weight + self.bias
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1 - self.p)
		input = input * bernoulli.sample(input.size()) * (1.0 / (1 - self.p))
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.linear1 = nn.Linear(3*32*32, 1024)
		self.bn1 = BatchNorm1d(num_features=1024)
		self.relu1 = torch.relu
		self.dropout = Dropout(drop_rate)
		self.linear2 = nn.Linear(1024, 512)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		#print(x.shape)
		l1 = self.linear1(x)
		b1 = self.bn1(l1)
		b1 = self.relu1(b1)
		b1 = self.dropout(b1)
		l2 = self.linear2(b1)
		logits = l2
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
