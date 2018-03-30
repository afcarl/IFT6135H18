import math
import torch
import torch.nn.functional as F
from controller import FeedForwardController, LSTMController
from head import Head
from torch import nn


def circular_conv(w, s):
    circular_w = torch.cat([w[:, -1].unsqueeze(1), w, w[:, 0].unsqueeze(1)], 1)
    ans = s[:, 2].unsqueeze(1) * circular_w[:, :-2]
    ans = ans + s[:, 1].unsqueeze(1) * circular_w[:, 1:-1]
    ans = ans + s[:, 0].unsqueeze(1) * circular_w[:, 2:]
    return ans


# TODO implement an LSTM controller

class NTM(nn.Module):

    def __init__(self, N, M, in_size, out_size, batch_size, lstm=False):
        super(NTM, self).__init__()

        if lstm:
            self.controller = LSTMController(in_size, out_size, M, batch_size)
        else:
            self.controller = FeedForwardController(in_size, out_size, M, batch_size)

        self.read_head = Head(batch_size, N, M)
        self.write_head = Head(batch_size, N, M)
        self.batch_size = batch_size
        self.N = N
        self.M = M
        self.eps = 1e-8
        self.memory = None
        self.register_parameter('memory_bias',
                                nn.Parameter(torch.randn(1, N, M) / math.sqrt(N)))

    def reset(self):
        self.controller.reset()
        self.memory = self.memory_bias.repeat(self.batch_size, 1, 1)
        self.write_head.reset()
        self.read_head.reset()

    def read(self):
        r = torch.bmm(self.read_head.attention.unsqueeze(1), self.memory)
        return r.view(self.batch_size, -1)

    def write(self, e, a):
        tilde_memory = self.memory * (1 - self.write_head.attention.unsqueeze(2) * e.unsqueeze(1))
        self.memory = tilde_memory + self.write_head.attention.unsqueeze(2) * a.unsqueeze(1)

    def addressing(self, param, head):
        k, beta, gate, shift, gamma = param
        # Content addressing
        K_matrix = F.cosine_similarity(k.unsqueeze(1), self.memory, dim=2)
        w_c = F.softmax(beta * K_matrix, dim=-1)

        # Interpolation
        w_g = gate * w_c + (1 - gate) * head.attention

        # Convolutional shift
        tilde_w = circular_conv(w_g, shift)

        # Sharpening
        tilde_sum = (tilde_w ** gamma).sum(1, keepdim=True)
        w_gamma = tilde_w ** gamma / (self.eps + tilde_sum)
        head.attention = w_gamma

    def send(self, x):
        h = self.controller(x)
        k, beta, g, s, gamma = self.write_head.compute_attention_params(h)
        self.addressing((k, beta, g, s, gamma), self.write_head)
        e, a = self.write_head.compute_write_params(h)
        self.write(e, a)

    def receive(self, x):
        h = self.controller(x)
        k, beta, gate, s, gamma = self.read_head.compute_attention_params(h)
        self.addressing((k, beta, gate, s, gamma), self.read_head)
        r = self.read()
        out = self.controller.compute_output(r)
        return out
