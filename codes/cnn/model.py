# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm2d(nn.Module):
    # TODO START
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features

        # Parameters
        shape = (1, num_features, 1, 1)
        self.weight = Parameter(torch.ones(shape))
        self.bias = Parameter(torch.zeros(shape))

        # Store the average mean and variance
        self.rm = torch.zeros(shape)
        self.rv = torch.ones(shape)
        self.register_buffer('running_mean', self.rm)
        self.register_buffer('running_var', self.rv)

        # Initialize your parameter
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
        input_mean = input.mean(axis=0)
        # print(input_mean.shape)
        input_var = input.var(axis=0)
        # print(input_var.shape)
        # print("rm",self.rm.shape,"innput", input_mean.shape)
        self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * input_mean
        self.running_var = (1-self.momentum) * self.running_var + self.momentum * input_var
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
        # input: [batch_size, num_feature_map, height, width]
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1 - self.p)
        input = input * bernoulli.sample(input.size()) * (1.0 / (1 - self.p))
        return input
# TODO END


class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = BatchNorm2d(num_features=32)
        self.relu1 = torch.relu
        self.dropout1 = Dropout(drop_rate)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = BatchNorm2d(num_features=64)
        self.relu2 = torch.relu
        self.dropout2 = Dropout(drop_rate)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64*6*6, 10)
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        #print(x.shape)
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        b1 = self.relu1(b1)
        b1 = self.dropout1(b1)
        b1 = self.pool1(b1)
        c2 = self.conv2(b1)
        b2 = self.bn2(c2)
        b2 = self.relu2(b2)
        b2 = self.dropout2(b2)
        b2 = self.pool2(b2)
        #print(b2.shape)
        b2 = b2.view(b2.size(0), -1)
        l = self.linear(b2)
        logits = l
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc


0
