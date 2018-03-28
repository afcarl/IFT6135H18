import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class Head(nn.Module):

    def __init__(self, batch_size, N, M):
        super(Head, self).__init__()
        self.beta_layer = nn.Linear(100, 1)
        self.gamma_layer = nn.Linear(100, 1)
        self.gate_layer = nn.Linear(100, 1)
        self.shift_layer = nn.Linear(100, 3)
        self.k_layer = nn.Linear(100, M)
        self.erase_layer = nn.Linear(100, M)
        self.add_layer = nn.Linear(100, M)
        self.attention = None
        self.attention_score_bias = Variable(
            torch.randn(1, N), requires_grad=True).cuda() / np.sqrt(N)
        self.batch_size = batch_size

    def compute_attention_params(self, h):
        beta = F.softplus(self.beta_layer(h))
        gamma = 1 + F.softplus(self.gamma_layer(h))
        k = self.k_layer(h)
        shift = F.softmax(self.shift_layer(h), dim=-1)
        gate = F.sigmoid(self.gate_layer(h))
        return k, beta, gate, shift, gamma

    def compute_write_params(self, h):
        e = F.sigmoid(self.erase_layer(h))
        a = self.add_layer(h)
        return e, a

    def reset(self):
        self.attention = F.softmax(self.attention_score_bias, \
                                   dim=1).repeat(self.batch_size, 1).clone()
