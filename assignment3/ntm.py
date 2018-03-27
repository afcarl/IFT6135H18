import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb
import numpy as np

def circular_conv(w, s):
    circular_w = torch.cat([w[:, -1].unsqueeze(1), w, w[:, 0].unsqueeze(1)], 1)
    ipdb.set_trace()
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c

class NTM(nn.Module):

    def __init__(self, N, M, in_size, out_size, batch_size, lstm=False):
        super(NTM, self).__init__()
        self.controller = Controller(in_size, M, out_size, lstm)
        self.read_head   = Head(batch_size, N, M)
        self.write_head = Head(batch_size, N, M)
        self.batch_size = batch_size
        self.N = N
        self.M = M
        self.eps = 1e-8
        self.memory = None
        self.memory_bias = Variable(torch.randn(batch_size, N, M).cuda(), requires_grad=True)


    def reset(self):
        self.memory = self.memory_bias
        self.write_head.reset()
        self.read_head.reset()

    def read(self):
        r = torch.bmm(self.read_head.attention.unsqueeze(1), self.memory)
        return r.squeeze()

    def write(self, e, a):
        tilde_memory = self.memory*(1-self.write_head.attention.unsqueeze(2)*e.unsqueeze(1))
        self.memory = tilde_memory + self.write_head.attention.unsqueeze(2)*a.unsqueeze(1)

    def addressing(self, param, head):
        k, beta, g, s, gamma = param
        # Content addressing
        assert k.size()[0] == self.batch_size
        assert k.size()[1] == self.M
        K_matrix = F.cosine_similarity(k.unsqueeze(1), self.memory, dim=2)
        assert K_matrix.size()[0] == self.batch_size
        assert K_matrix.size()[1] == self.N
        w_c = F.softmax(beta*K_matrix, dim=-1)

        # Interpolation
        w_g = g*w_c + (1-g)*head.attention

        # Convolutional shift
        tilde_w = circular_conv(w_g, s)
        #tilde_w = w_g

        # Sharpening
        tilde_sum = (tilde_w**gamma).sum(1, keepdim=True)
        w_gamma = tilde_w**gamma/(self.eps + tilde_sum)
        head.attention = w_gamma


    def send(self, x):
        h = self.controller(x)
        k, beta, g, s, gamma = self.write_head.compute_attention_params(h)
        self.addressing((k, beta, g, s, gamma), self.write_head)
        e, a = self.write_head.compute_write_params(h)
        self.write(e, a)

    def receive(self, x):
        h = self.controller(x)
        k, beta, g, s, gamma = self.read_head.compute_attention_params(h)
        self.addressing((k, beta, g, s, gamma), self.read_head)
        r = self.read()
        out = self.controller.compute_output(r)
        return out

class Head(nn.Module):

    def __init__(self, batch_size, N, M):
        super(Head, self).__init__()
        self.beta_layer = nn.Linear(100, 1)
        self.gamma_layer = nn.Linear(100, 1)
        self.g_layer = nn.Linear(100, 1)
        self.s_layer = nn.Linear(100, 3)
        self.k_layer = nn.Linear(100, M)
        self.e_layer = nn.Linear(100, M)
        self.a_layer = nn.Linear(100, M)
        self.attention = None
        self.attention_score_bias = Variable(torch.randn(batch_size,\
            N),requires_grad=True).cuda()

    def compute_attention_params(self, h):
        beta = F.softplus(self.beta_layer(h))
        gamma = 1 + F.softplus(self.gamma_layer(h))
        k = self.k_layer(h)
        s = self.s_layer(h)
        g = F.sigmoid(self.g_layer(h))
        return k, beta, g, s, gamma

    def compute_write_params(self, h):
        e = F.sigmoid(self.e_layer(h))
        a = self.a_layer(h)
        return e, a

    def reset(self):
        self.attention = F.softmax(self.attention_score_bias, dim=1)

class Controller(nn.Module):

    def __init__(self, in_size, M, out_size, lstm=False):
        super(Controller, self).__init__()
        self.layer = nn.Linear(in_size, 100)
        self.o_layer = nn.Linear(M, out_size)

    def forward(self, x):
        h = self.layer(x)
        return h

    def compute_output(self, r):
        return F.sigmoid(self.o_layer(r))

dim = 8
input = Variable(torch.bernoulli(torch.rand(32, 20, dim+1))).cuda()
input[:, :, -1] = 0
input[:, -1, -1] = 1

input_zero = Variable(torch.zeros(32, dim+1)).cuda()

M = 64
N = 512

ntm = NTM(N, M, dim+1, dim+1, batch_size=32)
ntm.cuda()

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(ntm.parameters(), lr=1e-3)


for epoch in range(10000):
    loss = 0
    ntm.reset()
    for i in range(20):
        ntm.send(input[:, i, :])

    for i in range(20):
        #x = ntm.receive()
        x = ntm.receive(input_zero)
        loss += criterion(x, input[:, i, :])

    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()





