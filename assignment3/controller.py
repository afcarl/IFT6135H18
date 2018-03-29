from torch import nn
from torch.nn import functional as F


class Controller(nn.Module):

    def __init__(self, in_size, M, out_size, lstm=False):
        super(Controller, self).__init__()
        self.layer = nn.Linear(in_size, 100)
        self.output_layer = nn.Linear(M, out_size)

    def forward(self, x):
        h = F.elu(self.layer(x))
        return h

    def compute_output(self, r):
        return F.sigmoid(self.output_layer(r))