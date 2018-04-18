import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid


class VAE(nn.Module):

    def __init__(self, input_size=784, hidden_size=500, encoding_size=2, activation='relu'):
        super(VAE, self).__init__()
        # ENCODER
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, encoding_size)  # mean of z|x
        self.fc22 = nn.Linear(hidden_size, encoding_size)  # std of z|x

        # DECODER
        self.fc3 = nn.Linear(encoding_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        self.encoding_size = encoding_size

        self.sigmoid = nn.Sigmoid()
        if activation == 'tanh':
            self.activation = nn.tanh()
        else:
            self.activation = nn.ReLU()

    def encode(self, x):
        # x's shape is (batch_size, input_size)
        h1 = self.activation(self.fc1(x))
        mu = self.fc21(h1)
        logsigma = self.fc22(h1)
        # mu and sigma's shape is (batch_size, encoding_size)
        return mu, logsigma  # we return logsigma to avoid adding a softplus layer

    def decode(self, z):
        h3 = self.activation(self.fc3(z))
        y = self.sigmoid(self.fc4(h3))
        # this is the output of the decoder (Bernoulli parameters)

        return y

    def loss(self, x, L=1):
        mu, logsigma = self.encode(x)
        sigma = logsigma.exp()

        loss_1 = .5 * (self.encoding_size + logsigma.mul(2).sum(dim=1) - mu.pow(2).sum(dim=1) - sigma.pow(2).sum(dim=1))
        # loss_1 is now a vector of size batch_size
        loss_1 = loss_1.mean()

        mu = mu.repeat(L, 1)
        sigma = sigma.repeat(L, 1)
        epsilon = Variable(mu.data.new(mu.shape).normal_())
        z = sigma * epsilon + mu
        y = self.decode(z)
        x_rep = x.repeat(L, 1)

        loss_2 = - F.binary_cross_entropy(y, x_rep) * x.shape[1]
        # print(loss_1, loss_2)
        return - (loss_1 + loss_2), y, - loss_2
