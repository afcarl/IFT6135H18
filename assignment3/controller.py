import torch
from torch import nn


class FeedForwardController(nn.Module):

    def __init__(self, in_size, out_size, M):
        super(FeedForwardController, self).__init__()
        self.in_net = nn.Linear(in_size, 100)
        self.out_net = nn.Linear(M, out_size)

    def forward(self, x):
        return self.in_net(x)

    def compute_output(self, read):
        return self.out_net(read)

    def reset(self):
        pass


class LSTMController(nn.Module):

    def __init__(self, in_size, out_size, M):
        super(LSTMController, self).__init__()
        self.lstm = nn.LSTMCell(in_size, 100)
        self.hidden_state = nn.Parameter(torch.zeros(1, 100).cuda())
        self.cell_state = nn.Parameter(torch.zeros(1, 100).cuda())
        self.out_net = nn.Linear(M, out_size)

    def forward(self, x):
        self.hidden_state, self.cell_state = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.hidden_state

    def compute_out(self, read):
        return self.out_net(read)

    def reset(self):
        self.hidden_state = nn.Parameter(torch.zeros(1, 100).cuda())
        self.cell_state = nn.Parameter(torch.zeros(1, 100).cuda())
