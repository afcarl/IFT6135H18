import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class FeedForwardController(nn.Module):

    def __init__(self, in_size, out_size, M, batch_size):
        super(FeedForwardController, self).__init__()
        self.in_net = nn.Linear(in_size, 100)
        self.out_net = nn.Linear(M, out_size)

    def forward(self, x):
        return F.elu(self.in_net(x))

    def compute_output(self, read):
        return self.out_net(read)

    def reset(self):
        pass


class LSTMController(nn.Module):

    def __init__(self, in_size, out_size, M, batch_size):
        super(LSTMController, self).__init__()

        self.batch_size = batch_size

        self.lstm = nn.LSTMCell(in_size, 100)
        self.hidden_bias = nn.Parameter(torch.zeros(1, 100))
        self.cell_bias = nn.Parameter(torch.zeros(1, 100))
        self.hidden_state = None
        self.cell_state = None

        self.out_net = nn.Linear(M, out_size)

    def forward(self, x):
        self.hidden_state, self.cell_state = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return F.elu(self.hidden_state)

    def compute_output(self, read):
        return self.out_net(read)

    def reset(self):
        self.hidden_state = self.hidden_bias.clone().repeat(self.batch_size, 1)
        self.cell_state = self.cell_bias.clone().repeat(self.batch_size, 1)
